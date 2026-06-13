import { spawn, type ChildProcess } from "node:child_process";

export class BackendSupervisor {
  private child?: ChildProcess;

  start() {
    if (this.child) return;
    const executable = process.env.DV_BACKEND_EXE ?? "python";
    const args = process.env.DV_BACKEND_EXE ? [] : ["-m", "dv_backend.main"];
    this.child = spawn(executable, args, { stdio: "pipe", windowsHide: true });
  }

  stop() {
    this.child?.kill();
    this.child = undefined;
  }
}

