import { app, BrowserWindow } from "electron";
import { BackendSupervisor } from "./backendSupervisor";

const backend = new BackendSupervisor();
app.whenReady().then(() => {
  backend.start();
  const window = new BrowserWindow({ width: 1280, height: 820, backgroundColor: "#0c0e13" });
  window.loadURL(process.env.VITE_DEV_SERVER_URL ?? `file://${app.getAppPath()}/dist/index.html`);
});
app.on("before-quit", () => backend.stop());

