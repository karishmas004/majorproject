import { useCallback, useRef, useState } from "react";
import { UploadCloud, FileWarning } from "lucide-react";

interface Props {
  accept: string;
  label: string;
  hint: string;
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export default function UploadDropzone({ accept, label, hint, onFileSelected, disabled }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0 || disabled) return;
      onFileSelected(files[0]);
    },
    [onFileSelected, disabled]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`relative flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-14 text-center transition-colors cursor-pointer
        ${isDragging ? "border-accent-cyan bg-[#E3EBE0]/70" : "border-[#DCE3DA] hover:border-[#A8B5AA]"}
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        disabled={disabled}
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div className="rounded-full bg-[#EEF2EB] p-4">
        <UploadCloud className="h-7 w-7 text-[#164D3B]" />
      </div>
      <p className="text-[#24352E] font-medium">{label}</p>
      <p className="text-[#707A74] text-sm flex items-center gap-1">
        <FileWarning className="h-3.5 w-3.5" /> {hint}
      </p>
    </div>
  );
}
