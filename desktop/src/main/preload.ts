import { contextBridge } from "electron";

contextBridge.exposeInMainWorld("runtime", { platform: process.platform });

