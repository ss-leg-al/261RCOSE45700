import { useRef, useState } from "react";

export default function EmptyView({ uploading, onUpload }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  function handleFile(file) {
    if (!file || !file.type.startsWith("video/")) return;
    onUpload(file);
  }

  if (uploading) {
    return (
      <div className="flex h-full items-center justify-center animate-fade-in">
        <div className="text-center">
          <div className="relative mx-auto mb-6 w-16 h-16">
            <div
              className="w-16 h-16 rounded-full animate-spin-slow"
              style={{ border: "2px solid transparent", borderTopColor: "#7c3aed", borderRightColor: "#3b82f6" }}
            />
            <div className="absolute inset-2 rounded-full" style={{ background: "var(--card)" }} />
          </div>
          <p className="font-semibold" style={{ color: "var(--text)" }}>업로드 중...</p>
          <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>잠시만 기다려주세요</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full items-center justify-center p-12 animate-fade-in">
      <div className="w-full max-w-lg text-center">

        {/* Hero text */}
        <div className="mb-10">
          <h1 className="text-4xl font-black mb-3">
            <span className="grad-text">영상 속 개인정보,</span>
            <br />
            <span style={{ color: "var(--text)" }}>AI가 알아서 처리합니다</span>
          </h1>
          <p className="text-sm leading-relaxed" style={{ color: "var(--muted)" }}>
            GPT-4o 씬 분석 · SAM3 픽셀 마스킹 · InsightFace 인물 식별<br />
            보호할 사람만 선택하면 나머지는 자동 블러
          </p>
        </div>

        {/* Drop zone */}
        <div
          className="rounded-2xl p-12 cursor-pointer transition-all"
          style={{
            background: dragging
              ? "linear-gradient(var(--card), var(--card)) padding-box, linear-gradient(135deg, #7c3aed, #3b82f6, #06b6d4) border-box"
              : "var(--card)",
            border: dragging ? "1px solid transparent" : "1px dashed rgba(255,255,255,0.12)",
            boxShadow: dragging ? "0 0 40px rgba(124,58,237,0.15)" : "none",
          }}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFile(e.dataTransfer.files[0]);
          }}
        >
          <div className="text-5xl mb-4">🎬</div>
          <p className="font-semibold mb-1" style={{ color: "var(--text)" }}>
            영상을 드래그하거나 클릭해서 선택
          </p>
          <p className="text-xs" style={{ color: "var(--muted)" }}>
            MP4, MOV, AVI — 최대 500MB
          </p>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {/* Feature chips */}
        <div className="flex flex-wrap justify-center gap-2 mt-6">
          {["SAM3 픽셀 마스킹", "GPT-4o 씬 분석", "인물 식별 보호", "감사 리포트"].map((f) => (
            <span
              key={f}
              className="text-xs px-3 py-1 rounded-full"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid var(--border)",
                color: "var(--muted)",
              }}
            >
              {f}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
