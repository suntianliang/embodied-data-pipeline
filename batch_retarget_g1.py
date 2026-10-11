#!/usr/bin/env python3
"""
batch_retarget_g1.py - Batch Retargeting SMPL-X / GVHMR Poses to Unitree G1 with MuJoCo Video Review

Usage:
    python batch_retarget_g1.py --input_dir data/poses_filtered --output_dir data/retargeted --video_dir data/videos_review --workers 4
"""

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
import numpy as np
import pickle

# Ensure GMR and project root are in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
GMR_ROOT = PROJECT_ROOT / "GMR"
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def process_single_clip(
    clip_pt_path: str,
    smplx_model_path: str,
    output_motion_dir: str,
    output_video_dir: str,
    robot_type: str = "unitree_g1",
    tgt_fps: int = 30,
    video_width: int = 640,
    video_height: int = 480,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Worker function to retarget a single clip to Unitree G1 and record MuJoCo review video.
    """
    clip_path = pathlib.Path(clip_pt_path)
    clip_id = clip_path.stem

    out_motion_pkl = pathlib.Path(output_motion_dir) / f"{clip_id}.pkl"
    out_motion_npz = pathlib.Path(output_motion_dir) / f"{clip_id}.npz"
    out_video_mp4 = pathlib.Path(output_video_dir) / f"{clip_id}.mp4"

    # Check if already processed
    if skip_existing and out_motion_pkl.exists() and out_video_mp4.exists() and out_video_mp4.stat().st_size > 1024:
        return {
            "clip_id": clip_id,
            "status": "SKIPPED_EXISTING",
            "motion_pkl": str(out_motion_pkl),
            "video_mp4": str(out_video_mp4),
            "video_size_bytes": out_video_mp4.stat().st_size,
        }

    start_time = time.time()

    try:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR
        from general_motion_retargeting import RobotMotionViewer
        from general_motion_retargeting.utils.smpl import (
            load_gvhmr_pred_file,
            get_gvhmr_data_offline_fast,
        )

        # 1. Load GVHMR predictions and SMPL-X forward kinematics
        smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(
            clip_path, pathlib.Path(smplx_model_path)
        )

        # 2. Extract aligned frames (30fps)
        smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
        )
        num_frames = len(smplx_data_frames)

        if num_frames == 0:
            return {
                "clip_id": clip_id,
                "status": "FAILED",
                "error": "0 frames extracted",
            }

        # 3. Initialize GMR IK Solver
        retarget = GMR(
            actual_human_height=human_height,
            src_human="smplx",
            tgt_robot=robot_type,
        )

        # 4. Solve IK for all frames
        qpos_list = []
        for frame_data in smplx_data_frames:
            qpos = retarget.retarget(frame_data)
            qpos_list.append(qpos)
        qpos_array = np.array(qpos_list, dtype=np.float32)

        # 5. GMR 侧三大关键后处理 (高度对齐 + 支撑相防滑步锁定 + 限位安全投影与平滑)
        from gmr_postprocess import GMRPostProcessor
        from robot_feasibility import RobotFeasibilityChecker

        postprocessor = GMRPostProcessor(max_joint_vel=25.0, smooth_cutoff_hz=6.0)
        qpos_array = postprocessor.process_all(qpos_array, fps=aligned_fps)

        # 6. Robot Feasibility Audit
        checker = RobotFeasibilityChecker()
        feasibility_res = checker.evaluate_feasibility(qpos_array, fps=aligned_fps)
        if not feasibility_res["passed"]:
            return {
                "clip_id": clip_id,
                "status": feasibility_res["status"],
                "reason": feasibility_res["reason"],
                "metrics": feasibility_res["metrics"],
                "num_frames": num_frames,
            }

        # 6. Initialize MuJoCo Review Video Recorder and render smoothed trajectory
        pathlib.Path(output_video_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_motion_dir).mkdir(parents=True, exist_ok=True)

        viewer = RobotMotionViewer(
            robot_type=robot_type,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=True,
            display=False,
            video_path=str(out_video_mp4),
            video_width=video_width,
            video_height=video_height,
        )

        for t in range(len(qpos_array)):
            qp = qpos_array[t]
            viewer.step(
                root_pos=qp[:3],
                root_rot=qp[3:7],
                dof_pos=qp[7:],
                rate_limit=False,
                follow_camera=True,
            )
        viewer.close()

        # 7. Format and save retargeted motion trajectories
        root_pos = qpos_array[:, :3]
        root_rot_wxyz = qpos_array[:, 3:7]
        # convert wxyz to xyzw for ROS/standard robotics formats
        root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
        dof_pos = qpos_array[:, 7:]

        dof_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        motion_data = {
            "clip_id": clip_id,
            "robot": robot_type,
            "fps": aligned_fps,
            "num_frames": num_frames,
            "duration_s": round(num_frames / aligned_fps, 3),
            "human_height": round(float(human_height), 3),
            "root_pos": root_pos,
            "root_rot": root_rot_xyzw,
            "root_rot_wxyz": root_rot_wxyz,
            "dof_pos": dof_pos,
            "qpos": qpos_array,
            "dof_names": dof_names,
        }

        with open(out_motion_pkl, "wb") as f:
            pickle.dump(motion_data, f)

        np.savez_compressed(
            out_motion_npz,
            qpos=qpos_array,
            root_pos=root_pos,
            root_rot_xyzw=root_rot_xyzw,
            root_rot_wxyz=root_rot_wxyz,
            dof_pos=dof_pos,
            fps=aligned_fps,
            num_frames=num_frames,
            human_height=float(human_height),
        )

        elapsed = time.time() - start_time
        fps_speed = num_frames / max(elapsed, 1e-4)

        return {
            "clip_id": clip_id,
            "status": "SUCCESS",
            "num_frames": num_frames,
            "duration_s": round(num_frames / aligned_fps, 2),
            "fps": aligned_fps,
            "processing_time_s": round(elapsed, 2),
            "speed_fps": round(fps_speed, 1),
            "motion_pkl": str(out_motion_pkl),
            "motion_npz": str(out_motion_npz),
            "video_mp4": str(out_video_mp4),
            "video_size_bytes": out_video_mp4.stat().st_size if out_video_mp4.exists() else 0,
        }

    except Exception as e:
        import traceback
        return {
            "clip_id": clip_id,
            "status": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description="Batch Retarget GVHMR Poses to Unitree G1 with MuJoCo Review")
    parser.add_argument("--input_dir", default="data/poses_filtered", help="Directory of filtered poses (.pt)")
    parser.add_argument("--output_dir", default="data/retargeted", help="Directory for retargeted motions")
    parser.add_argument("--video_dir", default="data/videos_review", help="Directory for MuJoCo review videos")
    parser.add_argument("--smplx_models", default="GMR/assets/body_models", help="SMPL-X body models dir")
    parser.add_argument("--robot", default="unitree_g1", choices=["unitree_g1", "unitree_g1_with_hands"])
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of clips to process (0 = all)")
    parser.add_argument("--no_skip", action="store_true", help="Force overwrite existing results")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_dir)
    pt_files = sorted(list(input_path.glob("*.pt")))
    if not pt_files:
        print(f"No .pt files found in {input_path}")
        sys.exit(1)

    if args.limit > 0:
        pt_files = pt_files[: args.limit]

    print(f"Found {len(pt_files)} qualified clips to retarget to {args.robot}.")
    print(f"Output motions -> {args.output_dir}")
    print(f"Output review videos -> {args.video_dir}")
    print(f"Workers: {args.workers}")

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path("reports").mkdir(parents=True, exist_ok=True)

    results = []
    start_total = time.time()
    success_count = 0
    fail_count = 0

    if args.workers == 1:
        # Sequential execution
        for idx, pt_file in enumerate(pt_files, 1):
            res = process_single_clip(
                str(pt_file),
                args.smplx_models,
                args.output_dir,
                args.video_dir,
                robot_type=args.robot,
                skip_existing=not args.no_skip,
            )
            results.append(res)
            if res["status"] in ("SUCCESS", "SKIPPED_EXISTING"):
                success_count += 1
                status_str = f"[{res['status']}] {res['clip_id']} ({res.get('num_frames', '?')} frames in {res.get('processing_time_s', '?')}s)"
            else:
                fail_count += 1
                status_str = f"[FAILED] {res['clip_id']}: {res.get('error')}"
            print(f"[{idx}/{len(pt_files)}] {status_str}")
    else:
        # Parallel execution with ProcessPoolExecutor
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            future_to_clip = {
                executor.submit(
                    process_single_clip,
                    str(pt_file),
                    args.smplx_models,
                    args.output_dir,
                    args.video_dir,
                    args.robot,
                    30,
                    640,
                    480,
                    not args.no_skip,
                ): pt_file
                for pt_file in pt_files
            }

            for idx, future in enumerate(as_completed(future_to_clip), 1):
                pt_file = future_to_clip[future]
                try:
                    res = future.result()
                    results.append(res)
                    if res["status"] in ("SUCCESS", "SKIPPED_EXISTING"):
                        success_count += 1
                        time_str = f"{res.get('processing_time_s', 0)}s ({res.get('speed_fps', 0)} fps)" if res["status"] == "SUCCESS" else "cached"
                        print(f"[{idx}/{len(pt_files)}] OK: {res['clip_id']} | {res.get('num_frames', 0)} frames | {time_str}")
                    else:
                        fail_count += 1
                        err_msg = res.get('reason') or res.get('error') or res.get('status')
                        print(f"[{idx}/{len(pt_files)}] 🚫 [{res['status']}] {res['clip_id']} | {err_msg}")
                except Exception as exc:
                    fail_count += 1
                    print(f"[{idx}/{len(pt_files)}] Worker Exception on {pt_file.stem}: {exc}")
                    results.append({"clip_id": pt_file.stem, "status": "FAILED", "error": str(exc)})

    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    print(f"Retargeting Finished!")
    print(f"Total processed: {len(pt_files)}")
    print(f"Success / Valid: {success_count} ({success_count / len(pt_files):.1%})")
    print(f"Failed: {fail_count}")
    print(f"Total Time: {total_time:.2f}s ({total_time / len(pt_files):.2f}s per clip)")
    print("=" * 60)

    # Save summary report
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "robot": args.robot,
        "total_clips": len(pt_files),
        "success_count": success_count,
        "failed_count": fail_count,
        "total_time_seconds": round(total_time, 2),
        "output_motion_dir": args.output_dir,
        "output_video_dir": args.video_dir,
        "records": results,
    }

    report_json = pathlib.Path("reports/retarget_summary.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary JSON saved to {report_json}")

    # Save CSV
    import csv
    report_csv = pathlib.Path("reports/retarget_summary.csv")
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "status", "num_frames", "duration_s", "processing_time_s", "speed_fps", "motion_pkl", "video_mp4", "error"])
        for r in results:
            writer.writerow([
                r.get("clip_id"),
                r.get("status"),
                r.get("num_frames", ""),
                r.get("duration_s", ""),
                r.get("processing_time_s", ""),
                r.get("speed_fps", ""),
                r.get("motion_pkl", ""),
                r.get("video_mp4", ""),
                r.get("error", ""),
            ])
    print(f"Summary CSV saved to {report_csv}")


if __name__ == "__main__":
    main()
