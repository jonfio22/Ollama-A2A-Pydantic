#!/usr/bin/env python3
"""Start all A2A agents in a single process using multiprocessing."""
import asyncio
import multiprocessing
import signal
import socket
import sys
import time
from typing import List

import uvicorn


def check_port(port: int) -> bool:
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def start_agent(app_name: str, port: int):
    """Start a single agent server."""
    print(f"Starting {app_name} on port {port}...")
    uvicorn.run(
        f"main:{app_name}",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


def main():
    """Start all agents in separate processes."""
    agents = [
        ("orchestrator_app", 8000),
        ("analyst_app", 8001),
        ("coder_app", 8002),
        ("validator_app", 8003),
    ]

    # Check for port conflicts
    print("Checking ports...")
    conflicts = []
    for app_name, port in agents:
        if not check_port(port):
            conflicts.append(port)

    if conflicts:
        print(f"\n❌ Error: The following ports are already in use: {', '.join(map(str, conflicts))}")
        print("\nTo fix this, run:")
        print(f"  lsof -ti:{','.join(map(str, conflicts))} | xargs kill -9")
        print("\nOr stop the processes using those ports.\n")
        sys.exit(1)

    processes: List[multiprocessing.Process] = []

    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nShutting down all agents...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)
        print("All agents stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start all agents
        for app_name, port in agents:
            p = multiprocessing.Process(
                target=start_agent,
                args=(app_name, port),
                name=f"{app_name}-{port}",
            )
            p.start()
            processes.append(p)
            time.sleep(1)  # Stagger startup

        print("\n" + "=" * 60)
        print("All agents started successfully!")
        print("=" * 60)
        print("\nAgent endpoints:")
        print("  - Orchestrator: http://localhost:8000")
        print("  - Analyst:      http://localhost:8001")
        print("  - Coder:        http://localhost:8002")
        print("  - Validator:    http://localhost:8003")
        print("\nPress Ctrl+C to stop all agents")
        print("=" * 60 + "\n")

        # Keep main process alive
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"Error: {e}")
        for p in processes:
            p.terminate()
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
