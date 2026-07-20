/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_HOSTED?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
