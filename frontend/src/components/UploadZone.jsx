import { useRef, useState } from "react";

export default function UploadZone({ onUpload }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  function handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      alert("영상 파일만 업로드할 수 있습니다.");
      return;
    }
    onUpload(file);
  }

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-6">
      <div className="w-full max-w-lg text-center">
        <h1 className="text-3xl font-bold text-white mb-2">SafeVlog</h1>
        <p className="text-gray-400 mb-10 text-sm">
          영상을 올리면 AI가 등장인물을 찾아줍니다.<br />
          지킬 사람만 선택하면 나머지는 자동으로 블러 처리됩니다.
        </p>

        <div
          className={`border-2 border-dashed rounded-2xl p-14 cursor-pointer transition-colors ${
            dragging
              ? "border-blue-400 bg-blue-950/30"
              : "border-gray-700 hover:border-gray-500 bg-gray-900"
          }`}
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFile(e.dataTransfer.files[0]);
          }}
        >
          <div className="text-5xl mb-4">🎬</div>
          <p className="text-gray-300 font-medium mb-1">영상을 드래그하거나 클릭해서 선택</p>
          <p className="text-gray-600 text-xs">MP4, MOV, AVI — 최대 500MB</p>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>
    </div>
  );
}
