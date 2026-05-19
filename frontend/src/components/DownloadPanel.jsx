import { api } from "../api/client";

export default function DownloadPanel({ jobId, onReset }) {
  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-6">
      <div className="w-full max-w-md text-center">
        <div className="text-6xl mb-6">✅</div>
        <h2 className="text-white text-2xl font-bold mb-2">처리 완료!</h2>
        <p className="text-gray-400 text-sm mb-8">
          선택한 인물 외 모든 얼굴이 블러 처리되었습니다.
        </p>

        <div className="flex flex-col gap-3">
          <a
            href={api.downloadUrl(jobId)}
            download="output.mp4"
            className="block bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 px-6 rounded-xl transition-colors"
          >
            영상 다운로드
          </a>
          <button
            onClick={onReset}
            className="text-gray-500 hover:text-gray-300 text-sm transition-colors py-2"
          >
            새 영상 처리하기
          </button>
        </div>
      </div>
    </div>
  );
}
