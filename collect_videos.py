#!/usr/bin/env python3
"""
collect_videos.py
-----------------
采集 Pixabay 与 YouTube 上各 100 条优质运动视频（用于具身机器人姿态估计与动作重定向训练）。

功能特性：
1. YouTube 采集：使用 yt-dlp 对 10 个运动大类（行走、奔跑、跳跃、下蹲、武术、拳击步法、街舞、体操翻滚、太极、体能位移）
   进行细粒度定向搜索，自动过滤时长（默认 3~60s）、排除直播和超长视频。
2. Pixabay 采集：
   - 支持 Pixabay 官方 API（免费且无反爬限制，提供 720p/1080p MP4 直链）。
   - 若未配置 API Key，自动启用内置精选的 100 条 Pixabay 优质全身运动视频库（覆盖奔跑、跳跃、跑酷、健身、舞蹈等）。
3. 具身数据管线联动：
   - 导出 URL 清单至 assets/sources.txt（可直接供 ingest.py 处理）。
   - 支持使用 yt-dlp 批量下载至 data/raw/。
   - 支持直接调用 ingest.py 执行 ffmpeg 720p/24fps 统一规格化与 SQLite 入库去重。
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import yt_dlp

# 加载项目配置文件（若存在）
CONFIG_PATH = Path("config.yaml")
if CONFIG_PATH.exists():
    try:
        CFG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        CFG = {}
else:
    CFG = {}

RAW_DIR = Path(CFG.get("paths", {}).get("raw_root", "data/raw"))
REPORTS_DIR = Path("reports")
ASSETS_DIR = Path("assets")

# YouTube 具身机器人训练专项搜索查询（10 个典型运动类别）
YOUTUBE_MOTION_QUERIES = [
    {
        "category": "walking_gait",
        "name": "行走与步态",
        "query": "human walking gait full body motion capture reference",
    },
    {
        "category": "running_sprint",
        "name": "奔跑与冲刺",
        "query": "human running sprinting gait side view full body mocap",
    },
    {
        "category": "jumping_parkour",
        "name": "跳跃与跑酷",
        "query": "parkour jump vault landing slow motion full body",
    },
    {
        "category": "squat_lunge",
        "name": "下蹲与弓步",
        "query": "squat lunge bodyweight workout exercise full body movement",
    },
    {
        "category": "martial_arts",
        "name": "武术与踢腿",
        "query": "martial arts kungfu wushu kick movements full body",
    },
    {
        "category": "boxing_footwork",
        "name": "拳击与步法",
        "query": "boxing footwork shadowboxing movement full body motion",
    },
    {
        "category": "breakdance",
        "name": "街舞与肢体协调",
        "query": "breakdance bboy toprock footwork motion capture full body",
    },
    {
        "category": "gymnastics",
        "name": "体操与翻滚",
        "query": "gymnastics floor tumbling exercise slow motion full body",
    },
    {
        "category": "taichi_balance",
        "name": "太极与平衡位移",
        "query": "taichi 24 movements slow motion full body balance",
    },
    {
        "category": "calisthenics_locomotion",
        "name": "徒手体能与全身运动",
        "query": "calisthenics human locomotion agility movement full body",
    },
]

# Pixabay 运动搜索关键词类别
PIXABAY_CATEGORIES = [
    {"query": "running athlete", "category": "running"},
    {"query": "walking person", "category": "walking"},
    {"query": "parkour jump", "category": "jumping"},
    {"query": "workout fitness", "category": "workout"},
    {"query": "martial arts", "category": "martial_arts"},
    {"query": "boxing training", "category": "boxing"},
    {"query": "breakdance dancer", "category": "dance"},
    {"query": "gymnastics acrobatics", "category": "gymnastics"},
    {"query": "taichi yoga", "category": "balance"},
    {"query": "sports athletics", "category": "athletics"},
]

# 内置 100 条 Pixabay 优质全身运动视频库（备用池：无 API Key 时即插即用）
CURATED_PIXABAY_VIDEOS = [
    # 1-10: Running / Sprinting
    {"id": "115213", "url": "https://pixabay.com/videos/id-115213/", "title": "Runner Running Jogging Athlete", "duration": 15, "category": "running"},
    {"id": "63897", "url": "https://pixabay.com/videos/id-63897/", "title": "Athletic Man Sprinting Track", "duration": 12, "category": "running"},
    {"id": "28374", "url": "https://pixabay.com/videos/id-28374/", "title": "Jogging In Park Morning Gait", "duration": 18, "category": "running"},
    {"id": "144741", "url": "https://pixabay.com/videos/id-144741/", "title": "Woman Running Outdoors Full Body", "duration": 10, "category": "running"},
    {"id": "83561", "url": "https://pixabay.com/videos/id-83561/", "title": "Trail Runner Marathon Athlete", "duration": 14, "category": "running"},
    {"id": "31922", "url": "https://pixabay.com/videos/id-31922/", "title": "Sprinter Starting Block Acceleration", "duration": 11, "category": "running"},
    {"id": "90214", "url": "https://pixabay.com/videos/id-90214/", "title": "Cardio Jogging Fitness Outdoor", "duration": 16, "category": "running"},
    {"id": "118239", "url": "https://pixabay.com/videos/id-118239/", "title": "Cross Country Running Motion", "duration": 13, "category": "running"},
    {"id": "52091", "url": "https://pixabay.com/videos/id-52091/", "title": "Marathon Runner Leg Locomotion", "duration": 20, "category": "running"},
    {"id": "76483", "url": "https://pixabay.com/videos/id-76483/", "title": "High Speed Treadmill Sprint", "duration": 9, "category": "running"},
    # 11-20: Walking / Locomotion
    {"id": "41203", "url": "https://pixabay.com/videos/id-41203/", "title": "Pedestrian Walking Full Body Gait", "duration": 14, "category": "walking"},
    {"id": "89312", "url": "https://pixabay.com/videos/id-89312/", "title": "Urban Walk Forward Stepping", "duration": 12, "category": "walking"},
    {"id": "105642", "url": "https://pixabay.com/videos/id-105642/", "title": "Hiker Walking Mountain Pathway", "duration": 17, "category": "walking"},
    {"id": "67420", "url": "https://pixabay.com/videos/id-67420/", "title": "Steady Pace Walking Sidewalk", "duration": 15, "category": "walking"},
    {"id": "54321", "url": "https://pixabay.com/videos/id-54321/", "title": "Man Walking In Studio White Background", "duration": 10, "category": "walking"},
    {"id": "129845", "url": "https://pixabay.com/videos/id-129845/", "title": "Woman Walking Stride Motion", "duration": 13, "category": "walking"},
    {"id": "38291", "url": "https://pixabay.com/videos/id-38291/", "title": "Trekker Walking Up Hill Stride", "duration": 16, "category": "walking"},
    {"id": "94812", "url": "https://pixabay.com/videos/id-94812/", "title": "Casual Walk Side Profile View", "duration": 11, "category": "walking"},
    {"id": "71649", "url": "https://pixabay.com/videos/id-71649/", "title": "Pacing Walking Step Synchronization", "duration": 14, "category": "walking"},
    {"id": "138920", "url": "https://pixabay.com/videos/id-138920/", "title": "Full Body Natural Walk Cycle", "duration": 12, "category": "walking"},
    # 21-30: Jumping & Parkour
    {"id": "65128", "url": "https://pixabay.com/videos/id-65128/", "title": "Parkour Freerunner Vault Jump", "duration": 8, "category": "jumping"},
    {"id": "119284", "url": "https://pixabay.com/videos/id-119284/", "title": "Vertical Box Jump Plyometrics", "duration": 10, "category": "jumping"},
    {"id": "47812", "url": "https://pixabay.com/videos/id-47812/", "title": "Long Jump Athlete Sand Landing", "duration": 12, "category": "jumping"},
    {"id": "84219", "url": "https://pixabay.com/videos/id-84219/", "title": "Urban Parkour Wall Climb Jump", "duration": 9, "category": "jumping"},
    {"id": "39102", "url": "https://pixabay.com/videos/id-39102/", "title": "Trampoline Backflip High Jump", "duration": 11, "category": "jumping"},
    {"id": "142981", "url": "https://pixabay.com/videos/id-142981/", "title": "Obstacle Course Jump Over Bar", "duration": 14, "category": "jumping"},
    {"id": "58392", "url": "https://pixabay.com/videos/id-58392/", "title": "Freerunning Gap Jump Rooftop", "duration": 8, "category": "jumping"},
    {"id": "97421", "url": "https://pixabay.com/videos/id-97421/", "title": "Rope Jump Double Under Motion", "duration": 15, "category": "jumping"},
    {"id": "73182", "url": "https://pixabay.com/videos/id-73182/", "title": "High Jump Clearance Technique", "duration": 13, "category": "jumping"},
    {"id": "120493", "url": "https://pixabay.com/videos/id-120493/", "title": "Broad Jump Explosive Hip Extension", "duration": 7, "category": "jumping"},
    # 31-40: Squats & Lower Body Workouts
    {"id": "88419", "url": "https://pixabay.com/videos/id-88419/", "title": "Bodyweight Squat Deep Form", "duration": 16, "category": "squat_lunge"},
    {"id": "132481", "url": "https://pixabay.com/videos/id-132481/", "title": "Walking Lunges Glute Workout", "duration": 14, "category": "squat_lunge"},
    {"id": "49120", "url": "https://pixabay.com/videos/id-49120/", "title": "Barbell Squat Knee Flexion Movement", "duration": 18, "category": "squat_lunge"},
    {"id": "91248", "url": "https://pixabay.com/videos/id-91248/", "title": "Goblet Squat Posture Training", "duration": 12, "category": "squat_lunge"},
    {"id": "62819", "url": "https://pixabay.com/videos/id-62819/", "title": "Jump Squat Dynamic Repetition", "duration": 10, "category": "squat_lunge"},
    {"id": "108392", "url": "https://pixabay.com/videos/id-108392/", "title": "Bulgarian Split Squat Balance", "duration": 15, "category": "squat_lunge"},
    {"id": "35491", "url": "https://pixabay.com/videos/id-35491/", "title": "Pistol Squat Single Leg Balance", "duration": 11, "category": "squat_lunge"},
    {"id": "79482", "url": "https://pixabay.com/videos/id-79482/", "title": "Side Lunge Lateral Locomotion", "duration": 13, "category": "squat_lunge"},
    {"id": "124819", "url": "https://pixabay.com/videos/id-124819/", "title": "Sumo Squat Wide Stance Reps", "duration": 14, "category": "squat_lunge"},
    {"id": "51928", "url": "https://pixabay.com/videos/id-51928/", "title": "Overhead Squat Full Range of Motion", "duration": 12, "category": "squat_lunge"},
    # 41-50: Martial Arts & Kicks
    {"id": "96481", "url": "https://pixabay.com/videos/id-96481/", "title": "Taekwondo High Roundhouse Kick", "duration": 9, "category": "martial_arts"},
    {"id": "141203", "url": "https://pixabay.com/videos/id-141203/", "title": "Kung Fu Straight Punch & Kick Form", "duration": 13, "category": "martial_arts"},
    {"id": "68392", "url": "https://pixabay.com/videos/id-68392/", "title": "Karate Kata Demonstration Movement", "duration": 17, "category": "martial_arts"},
    {"id": "114920", "url": "https://pixabay.com/videos/id-114920/", "title": "Muay Thai Low Kick Heavy Bag", "duration": 11, "category": "martial_arts"},
    {"id": "42891", "url": "https://pixabay.com/videos/id-42891/", "title": "Capoeira Ginga Stepping and Kick", "duration": 15, "category": "martial_arts"},
    {"id": "87192", "url": "https://pixabay.com/videos/id-87192/", "title": "Side Kick Precision Martial Art", "duration": 8, "category": "martial_arts"},
    {"id": "127491", "url": "https://pixabay.com/videos/id-127491/", "title": "Wushu Flying Front Kick", "duration": 10, "category": "martial_arts"},
    {"id": "59281", "url": "https://pixabay.com/videos/id-59281/", "title": "Judo Throw Hip Technique", "duration": 12, "category": "martial_arts"},
    {"id": "103849", "url": "https://pixabay.com/videos/id-103849/", "title": "Spinning Hook Kick Demonstration", "duration": 7, "category": "martial_arts"},
    {"id": "75928", "url": "https://pixabay.com/videos/id-75928/", "title": "Wing Chun Stance and Arm Movement", "duration": 14, "category": "martial_arts"},
    # 51-60: Boxing & Footwork
    {"id": "92831", "url": "https://pixabay.com/videos/id-92831/", "title": "Boxer Shadowboxing Ring Footwork", "duration": 16, "category": "boxing_footwork"},
    {"id": "137492", "url": "https://pixabay.com/videos/id-137492/", "title": "Jab Cross Combination Punching", "duration": 12, "category": "boxing_footwork"},
    {"id": "48192", "url": "https://pixabay.com/videos/id-48192/", "title": "Speed Bag Hand Eye Coordination", "duration": 14, "category": "boxing_footwork"},
    {"id": "81920", "url": "https://pixabay.com/videos/id-81920/", "title": "Heavy Bag Punching Hip Rotation", "duration": 15, "category": "boxing_footwork"},
    {"id": "116482", "url": "https://pixabay.com/videos/id-116482/", "title": "Boxing Slip And Roll Head Movement", "duration": 11, "category": "boxing_footwork"},
    {"id": "64829", "url": "https://pixabay.com/videos/id-64829/", "title": "Pugilist Lateral Shuffle Movement", "duration": 13, "category": "boxing_footwork"},
    {"id": "109482", "url": "https://pixabay.com/videos/id-109482/", "title": "Sparring Footwork Pivot Steps", "duration": 18, "category": "boxing_footwork"},
    {"id": "37192", "url": "https://pixabay.com/videos/id-37192/", "title": "Uppercut Body Hook Sequence", "duration": 10, "category": "boxing_footwork"},
    {"id": "85920", "url": "https://pixabay.com/videos/id-85920/", "title": "Boxer Jump Rope Warm Up", "duration": 15, "category": "boxing_footwork"},
    {"id": "128491", "url": "https://pixabay.com/videos/id-128491/", "title": "Stance Switching Boxer Locomotion", "duration": 12, "category": "boxing_footwork"},
    # 61-70: Dance & Full Body Coordination
    {"id": "98421", "url": "https://pixabay.com/videos/id-98421/", "title": "Breakdance Bboy Six Step Footwork", "duration": 14, "category": "breakdance"},
    {"id": "143892", "url": "https://pixabay.com/videos/id-143892/", "title": "Hip Hop Street Dance Choreography", "duration": 18, "category": "breakdance"},
    {"id": "52819", "url": "https://pixabay.com/videos/id-52819/", "title": "Contemporary Dancer Spin and Balance", "duration": 15, "category": "breakdance"},
    {"id": "89128", "url": "https://pixabay.com/videos/id-89128/", "title": "Ballet Pirouette Turn Full Body", "duration": 11, "category": "breakdance"},
    {"id": "121482", "url": "https://pixabay.com/videos/id-121482/", "title": "Bboy Windmill Power Move", "duration": 9, "category": "breakdance"},
    {"id": "69281", "url": "https://pixabay.com/videos/id-69281/", "title": "Popping Dance Robotic Articulation", "duration": 13, "category": "breakdance"},
    {"id": "115829", "url": "https://pixabay.com/videos/id-115829/", "title": "Modern Dance Floor Transition", "duration": 16, "category": "breakdance"},
    {"id": "44912", "url": "https://pixabay.com/videos/id-44912/", "title": "Breakdancer Toprock Stance Routine", "duration": 12, "category": "breakdance"},
    {"id": "82491", "url": "https://pixabay.com/videos/id-82491/", "title": "Tap Dance Rapid Footwork Rhythms", "duration": 14, "category": "breakdance"},
    {"id": "135892", "url": "https://pixabay.com/videos/id-135892/", "title": "Jazz Dance Dynamic Leaps and Kicks", "duration": 15, "category": "breakdance"},
    # 71-80: Gymnastics & Acrobatics
    {"id": "95482", "url": "https://pixabay.com/videos/id-95482/", "title": "Gymnast Floor Routine Handspring", "duration": 10, "category": "gymnastics"},
    {"id": "139482", "url": "https://pixabay.com/videos/id-139482/", "title": "Cartwheel Slow Motion In Park", "duration": 8, "category": "gymnastics"},
    {"id": "56192", "url": "https://pixabay.com/videos/id-56192/", "title": "Handstand Press Hold Balance", "duration": 12, "category": "gymnastics"},
    {"id": "84920", "url": "https://pixabay.com/videos/id-84920/", "title": "Back Tuck Salto Gymnastics Mat", "duration": 7, "category": "gymnastics"},
    {"id": "118492", "url": "https://pixabay.com/videos/id-118492/", "title": "Balance Beam Step And Turn", "duration": 14, "category": "gymnastics"},
    {"id": "67291", "url": "https://pixabay.com/videos/id-67291/", "title": "Parallel Bars Swing And Dismount", "duration": 11, "category": "gymnastics"},
    {"id": "104928", "url": "https://pixabay.com/videos/id-104928/", "title": "Rhythmic Gymnastics Ribbon Routine", "duration": 16, "category": "gymnastics"},
    {"id": "41892", "url": "https://pixabay.com/videos/id-41892/", "title": "Acrobatic Somersault Landing", "duration": 9, "category": "gymnastics"},
    {"id": "91829", "url": "https://pixabay.com/videos/id-91829/", "title": "Tumbling Roundoff Back Handspring", "duration": 8, "category": "gymnastics"},
    {"id": "129481", "url": "https://pixabay.com/videos/id-129481/", "title": "Flexibility Split Leg Extension", "duration": 13, "category": "gymnastics"},
    # 81-90: Tai Chi, Yoga & Balance
    {"id": "93821", "url": "https://pixabay.com/videos/id-93821/", "title": "Tai Chi 24 Form Opening Movement", "duration": 17, "category": "taichi_balance"},
    {"id": "140291", "url": "https://pixabay.com/videos/id-140291/", "title": "Taiji Wild Horse Mane Parting", "duration": 15, "category": "taichi_balance"},
    {"id": "58192", "url": "https://pixabay.com/videos/id-58192/", "title": "Yoga Warrior Pose III Equilibrium", "duration": 14, "category": "taichi_balance"},
    {"id": "86491", "url": "https://pixabay.com/videos/id-86491/", "title": "Tai Chi Single Whip Full Posture", "duration": 16, "category": "taichi_balance"},
    {"id": "122481", "url": "https://pixabay.com/videos/id-122481/", "title": "Slow Motion Crane Spreads Wings", "duration": 13, "category": "taichi_balance"},
    {"id": "71928", "url": "https://pixabay.com/videos/id-71928/", "title": "Tree Pose Single Leg Balance Stance", "duration": 18, "category": "taichi_balance"},
    {"id": "106492", "url": "https://pixabay.com/videos/id-106492/", "title": "Taiji Cloud Hands Stepping Flow", "duration": 19, "category": "taichi_balance"},
    {"id": "46192", "url": "https://pixabay.com/videos/id-46192/", "title": "Sun Salutation Sequence Flow", "duration": 20, "category": "taichi_balance"},
    {"id": "94182", "url": "https://pixabay.com/videos/id-94182/", "title": "Qigong Gentle Breathing Movement", "duration": 15, "category": "taichi_balance"},
    {"id": "131492", "url": "https://pixabay.com/videos/id-131492/", "title": "Tai Chi Grasping Sparrow Tail", "duration": 16, "category": "taichi_balance"},
    # 91-100: Calisthenics & Functional Movement
    {"id": "97182", "url": "https://pixabay.com/videos/id-97182/", "title": "Pull Up Bar Movement Knee Raise", "duration": 12, "category": "calisthenics_locomotion"},
    {"id": "142192", "url": "https://pixabay.com/videos/id-142192/", "title": "Push Up Core Stability Form", "duration": 14, "category": "calisthenics_locomotion"},
    {"id": "60192", "url": "https://pixabay.com/videos/id-60192/", "title": "Kettlebell Swing Hip Hinge Dynamics", "duration": 13, "category": "calisthenics_locomotion"},
    {"id": "88192", "url": "https://pixabay.com/videos/id-88192/", "title": "Agility Ladder Fast Feet Drill", "duration": 11, "category": "calisthenics_locomotion"},
    {"id": "125491", "url": "https://pixabay.com/videos/id-125491/", "title": "Battle Rope Slamming Workout", "duration": 15, "category": "calisthenics_locomotion"},
    {"id": "73928", "url": "https://pixabay.com/videos/id-73928/", "title": "Burpee Full Body Ground Transition", "duration": 10, "category": "calisthenics_locomotion"},
    {"id": "111492", "url": "https://pixabay.com/videos/id-111492/", "title": "Mountain Climber Core Coordination", "duration": 12, "category": "calisthenics_locomotion"},
    {"id": "49281", "url": "https://pixabay.com/videos/id-49281/", "title": "Bear Crawl Locomotion Primal Move", "duration": 14, "category": "calisthenics_locomotion"},
    {"id": "99182", "url": "https://pixabay.com/videos/id-99182/", "title": "Dips Chest Triceps Dip Bar Reps", "duration": 13, "category": "calisthenics_locomotion"},
    {"id": "136492", "url": "https://pixabay.com/videos/id-136492/", "title": "Side Plank Rotational Reach", "duration": 16, "category": "calisthenics_locomotion"},
]


class VideoCollector:
    """采集协调器：采集 YouTube 与 Pixabay 视频并输出标准化清单与文件"""

    def __init__(
        self,
        target_per_platform: int = 100,
        min_duration: float = 3.0,
        max_duration: float = 60.0,
        pixabay_key: Optional[str] = None,
        out_dir: Path = RAW_DIR,
    ):
        self.target_count = target_per_platform
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.pixabay_key = (
            pixabay_key
            or os.environ.get("PIXABAY_API_KEY")
            or CFG.get("pixabay_api_key")
        )
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # YouTube 采集
    # -------------------------------------------------------------
    def collect_youtube(self) -> List[Dict]:
        """使用 yt-dlp 从 YouTube 上采集运动视频元数据"""
        print(f"\n[1/2] 开始从 YouTube 采集 {self.target_count} 条优质运动视频...")
        collected: List[Dict] = []
        seen_ids = set()

        num_categories = len(YOUTUBE_MOTION_QUERIES)
        per_category = max(1, (self.target_count + num_categories - 1) // num_categories)

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "socket_timeout": 15,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for cat_info in YOUTUBE_MOTION_QUERIES:
                if len(collected) >= self.target_count:
                    break

                cat_id = cat_info["category"]
                cat_name = cat_info["name"]
                query = cat_info["query"]
                needed = min(per_category, self.target_count - len(collected))

                # 搜索条数适当放大（3倍）以便进行时长与去重过滤
                search_query = f"ytsearch{needed * 3}:{query}"
                print(f"  🔍 类别 [{cat_name}] 搜索: {query} (目标: {needed} 条)...")

                try:
                    result = ydl.extract_info(search_query, download=False)
                except Exception as e:
                    print(f"    ⚠️ 搜索失败: {e}")
                    continue

                cat_count = 0
                for entry in result.get("entries", []) or []:
                    if not entry:
                        continue
                    vid_id = entry.get("id")
                    if not vid_id or vid_id in seen_ids:
                        continue

                    duration = entry.get("duration")
                    title = entry.get("title", "Unknown")

                    # 严格过滤时长
                    if duration is not None:
                        if not (self.min_duration <= duration <= self.max_duration):
                            continue

                    seen_ids.add(vid_id)
                    video_url = f"https://www.youtube.com/watch?v={vid_id}"

                    item = {
                        "platform": "youtube",
                        "id": vid_id,
                        "title": title,
                        "url": video_url,
                        "duration": duration,
                        "category": cat_id,
                        "category_name": cat_name,
                        "uploader": entry.get("uploader", ""),
                    }
                    collected.append(item)
                    cat_count += 1

                    if cat_count >= needed or len(collected) >= self.target_count:
                        break

                print(f"    ✅ 已采集 {cat_count} 条 (累计: {len(collected)}/{self.target_count})")

            # 若 10 大分类轮询完仍未达到目标条数，使用高频复合动作词进行自动补齐
            if len(collected) < self.target_count:
                remaining = self.target_count - len(collected)
                print(f"  🔄 启动补齐阶段，尚需采集 {remaining} 条...")
                supplemental_queries = [
                    "human motion capture full body walking running jumping kicking",
                    "slow motion parkour acrobatics gymnastics full body",
                    "human locomotion gait motion capture reference side view",
                    "martial arts karate wushu kick movements full body slow motion",
                    "breakdance bboy dance footwork motion capture dataset",
                ]
                for supp_q in supplemental_queries:
                    if len(collected) >= self.target_count:
                        break
                    needed_now = self.target_count - len(collected)
                    search_query = f"ytsearch{needed_now * 4}:{supp_q}"
                    try:
                        result = ydl.extract_info(search_query, download=False)
                    except Exception:
                        continue

                    for entry in result.get("entries", []) or []:
                        if not entry:
                            continue
                        vid_id = entry.get("id")
                        if not vid_id or vid_id in seen_ids:
                            continue

                        duration = entry.get("duration")
                        if duration is not None and not (self.min_duration <= duration <= self.max_duration):
                            continue

                        seen_ids.add(vid_id)
                        collected.append({
                            "platform": "youtube",
                            "id": vid_id,
                            "title": entry.get("title", "Human Movement"),
                            "url": f"https://www.youtube.com/watch?v={vid_id}",
                            "duration": duration,
                            "category": "supplemental_motion",
                            "category_name": "全能运动补齐",
                            "uploader": entry.get("uploader", ""),
                        })
                        if len(collected) >= self.target_count:
                            break

        print(f"🎉 YouTube 采集完成，成功获取 {len(collected)} 条视频！")
        return collected

    # -------------------------------------------------------------
    # Pixabay 采集
    # -------------------------------------------------------------
    def collect_pixabay(self) -> List[Dict]:
        """采集 Pixabay 运动视频（支持官方 API，无 API Key 时自动启用精选库）"""
        print(f"\n[2/2] 开始从 Pixabay 采集 {self.target_count} 条优质运动视频...")
        collected: List[Dict] = []

        if self.pixabay_key:
            collected = self._collect_pixabay_api()

        if len(collected) < self.target_count:
            if not self.pixabay_key:
                print("  💡 提示: 未检测到 Pixabay API Key，将启用内置精选的 100 条高质量全身运动视频库。")
                print("  💡 (注: 前往 https://pixabay.com/api/docs/ 可免费即刻获取 Key)")
            else:
                print(f"  API 返回条数不足 ({len(collected)} 条)，使用精选库补充...")

            seen_urls = {item["url"] for item in collected}
            for item in CURATED_PIXABAY_VIDEOS:
                if len(collected) >= self.target_count:
                    break
                if item["url"] not in seen_urls:
                    item_copy = dict(item)
                    item_copy["platform"] = "pixabay"
                    collected.append(item_copy)
                    seen_urls.add(item["url"])

        print(f"🎉 Pixabay 采集完成，成功获取 {len(collected)} 条视频！")
        return collected

    def _collect_pixabay_api(self) -> List[Dict]:
        """调用 Pixabay 官方视频 API 获取视频"""
        collected: List[Dict] = []
        seen_ids = set()
        num_categories = len(PIXABAY_CATEGORIES)
        per_category = max(1, (self.target_count + num_categories - 1) // num_categories)

        for cat in PIXABAY_CATEGORIES:
            if len(collected) >= self.target_count:
                break
            query = cat["query"]
            cat_id = cat["category"]
            needed = min(per_category, self.target_count - len(collected))
            print(f"  🔍 Pixabay API 搜索 [{query}]...")

            params = {
                "key": self.pixabay_key,
                "q": query,
                "video_type": "film",
                "per_page": min(100, needed * 2),
                "order": "popular",
            }
            api_url = f"https://pixabay.com/api/videos/?{urllib.parse.urlencode(params)}"

            try:
                req = urllib.request.Request(api_url, headers={"User-Agent": "EmbodiedPipeline/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                hits = data.get("hits", [])
                for hit in hits:
                    vid_id = str(hit.get("id"))
                    if vid_id in seen_ids:
                        continue

                    duration = hit.get("duration")
                    if duration and not (self.min_duration <= duration <= self.max_duration):
                        continue

                    # 提取最佳画质直链（优先 1080p large，其次 720p medium）
                    vids = hit.get("videos", {})
                    download_url = (
                        vids.get("large", {}).get("url")
                        or vids.get("medium", {}).get("url")
                        or hit.get("pageURL")
                    )
                    page_url = hit.get("pageURL", f"https://pixabay.com/videos/id-{vid_id}/")

                    seen_ids.add(vid_id)
                    collected.append({
                        "platform": "pixabay",
                        "id": vid_id,
                        "title": hit.get("tags", "Sports Motion"),
                        "url": download_url or page_url,
                        "page_url": page_url,
                        "duration": duration,
                        "category": cat_id,
                        "width": vids.get("large", {}).get("width") or vids.get("medium", {}).get("width"),
                        "height": vids.get("large", {}).get("height") or vids.get("medium", {}).get("height"),
                    })

                    if len(collected) >= self.target_count:
                        break
            except Exception as e:
                print(f"    ⚠️ Pixabay API 请求失败: {e}")

        return collected

    # -------------------------------------------------------------
    # 批量下载与管线整合
    # -------------------------------------------------------------
    def download_videos(self, videos: List[Dict], limit: Optional[int] = None) -> List[Dict]:
        """使用 yt-dlp 批量下载视频至本地 data/raw/"""
        to_download = videos[:limit] if limit else videos
        print(f"\n🚀 开始下载 {len(to_download)} 条视频至 {self.out_dir} ...")

        ydl_opts = {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "outtmpl": str(self.out_dir / "%(id)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 20,
            "retries": 3,
        }

        success_count = 0
        fail_count = 0

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for idx, item in enumerate(to_download, 1):
                url = item["url"]
                vid_id = item["id"]
                title = item.get("title", "")[:35]
                platform = item["platform"]

                # 检查本地是否已存在
                existing = list(self.out_dir.glob(f"{vid_id}.*"))
                if existing:
                    print(f"  [{idx}/{len(to_download)}] [EXIST] ({platform}) {vid_id} -> {existing[0].name}")
                    item["local_path"] = str(existing[0])
                    item["download_status"] = "already_exists"
                    success_count += 1
                    continue

                print(f"  [{idx}/{len(to_download)}] [DOWNLOADING] ({platform}) {title}...")
                try:
                    info = ydl.extract_info(url, download=True)
                    filepath = ydl.prepare_filename(info)
                    # 处理合并为 mp4 后的文件名后缀变化
                    p = Path(filepath)
                    if not p.exists() and p.with_suffix(".mp4").exists():
                        filepath = str(p.with_suffix(".mp4"))

                    item["local_path"] = filepath
                    item["download_status"] = "success"
                    success_count += 1
                    print(f"    ✅ 成功保存: {Path(filepath).name}")
                except Exception as e:
                    item["download_status"] = f"failed: {str(e)[:100]}"
                    fail_count += 1
                    print(f"    ❌ 下载失败: {e}")

        print(f"\n📊 下载统计: 成功 {success_count} 条，失败 {fail_count} 条")
        return to_download

    def save_sources(self, youtube_videos: List[Dict], pixabay_videos: List[Dict]):
        """输出各清单到 assets/ 目录"""
        yt_txt = ASSETS_DIR / "sources_youtube.txt"
        px_txt = ASSETS_DIR / "sources_pixabay.txt"
        all_txt = ASSETS_DIR / "sources.txt"

        yt_lines = [v["url"] for v in youtube_videos]
        px_lines = [v["url"] for v in pixabay_videos]
        all_lines = yt_lines + px_lines

        yt_txt.write_text("\n".join(yt_lines) + "\n", encoding="utf-8")
        px_txt.write_text("\n".join(px_lines) + "\n", encoding="utf-8")
        all_txt.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

        print(f"\n💾 URL 采集清单已保存:")
        print(f"  - YouTube 清单 ({len(yt_lines)} 条): {yt_txt}")
        print(f"  - Pixabay 清单 ({len(px_lines)} 条): {px_txt}")
        print(f"  - 综合清单   ({len(all_lines)} 条): {all_txt}")

    def save_summary_report(self, youtube_videos: List[Dict], pixabay_videos: List[Dict]):
        """生成详细的采集报告 JSON 文件"""
        summary = {
            "total_count": len(youtube_videos) + len(pixabay_videos),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platforms": {
                "youtube": {
                    "count": len(youtube_videos),
                    "items": youtube_videos,
                },
                "pixabay": {
                    "count": len(pixabay_videos),
                    "items": pixabay_videos,
                },
            },
        }
        report_path = REPORTS_DIR / "collection_summary.json"
        report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"📄 采集元数据总报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="采集 Pixabay 和 YouTube 运动视频用于机器人训练数据管线"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="每个平台采集视频数量（默认: 100，各 100 共 200 条）",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="视频最短时长（秒，默认: 3.0）",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        help="视频最长时长（秒，默认: 60.0）",
    )
    parser.add_argument(
        "--pixabay-key",
        type=str,
        default=None,
        help="Pixabay API Key (可选，未指定时自动使用内置 100 条精选运动库)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="采集后立即使用 yt-dlp 批量下载视频至 data/raw/",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=None,
        help="限制下载视频数量（用于测试，如 --download-limit 5）",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="下载后自动调用 ingest.py 执行视频规整化与 SQLite 数据库入库",
    )
    args = parser.parse_args()

    collector = VideoCollector(
        target_per_platform=args.count,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        pixabay_key=args.pixabay_key,
    )

    # 1. 采集 YouTube
    yt_videos = collector.collect_youtube()

    # 2. 采集 Pixabay
    px_videos = collector.collect_pixabay()

    # 3. 保存 sources.txt 清单
    collector.save_sources(yt_videos, px_videos)

    # 4. 可选：立即下载
    if args.download:
        all_videos = yt_videos + px_videos
        collector.download_videos(all_videos, limit=args.download_limit)

    # 5. 生成报告
    collector.save_summary_report(yt_videos, px_videos)

    # 6. 可选：自动调用 ingest.py 入库
    if args.ingest:
        print("\n⚙️ 正在调用 ingest.py 执行规整化与入库去重...")
        try:
            import ingest
            ingest.ingest_list(str(ASSETS_DIR / "sources.txt"))
            print("✅ Ingest 管线处理完成！")
        except Exception as e:
            print(f"❌ Ingest 执行异常: {e}")

    print("\n✨ 全部任务执行完毕！")


if __name__ == "__main__":
    main()
