import { useState } from "react";
import { ImageIcon, Video, AudioLines, Camera, ShieldCheck } from "lucide-react";
import ImageDetection from "./components/ImageDetection";
import VideoDetection from "./components/VideoDetection";
import AudioDetection from "./components/AudioDetection";
import WebcamDetection from "./components/WebcamDetection";

type Tab = "image" | "video" | "audio" | "webcam";

const TABS: { id: Tab; label: string; icon: typeof ImageIcon }[] = [
  { id: "image", label: "Image", icon: ImageIcon },
  { id: "video", label: "Video", icon: Video },
  { id: "audio", label: "Audio", icon: AudioLines },
  { id: "webcam", label: "Live Webcam", icon: Camera },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("image");

  return (
    <div className="min-h-screen bg-bg">
      <header className="border-b border-[#DCE3DA] bg-bg/80 backdrop-blur sticky top-0 z-10">
        <div className="mx-auto max-w-6xl px-6 py-4 flex items-center gap-3">
          <div className="rounded-xl bg-[#E3EBE0] p-2">
            <ShieldCheck className="h-6 w-6 text-[#164D3B]" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-[#0F1F2A]">Multimodal Deepfake Detection System</h1>
            <p className="text-xs text-[#707A74]">Image · Video · Audio · Live Webcam — EfficientNetV2 + CNN Spectrogram, Grad-CAM explainability</p>
          </div>
        </div>
      </header>

      <nav className="mx-auto max-w-6xl px-6 mt-6">
        <div className="flex gap-2 rounded-2xl bg-bg-card border border-[#DCE3DA] p-1.5 w-fit">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition-colors ${
                tab === id ? "bg-[#164D3B] text-white" : "text-[#5F6868] hover:text-[#24352E]"
              }`}
            >
              <Icon className="h-4 w-4" /> {label}
            </button>
          ))}
        </div>
      </nav>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {tab === "image" && <ImageDetection />}
        {tab === "video" && <VideoDetection />}
        {tab === "audio" && <AudioDetection />}
        {tab === "webcam" && <WebcamDetection />}
      </main>

      <footer className="mx-auto max-w-6xl px-6 py-8 text-xs text-[#7B857F]">
        Runs entirely on your own backend — no data leaves your machine.
      </footer>
    </div>
  );
}
