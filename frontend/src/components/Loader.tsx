import { Loader2 } from "lucide-react";

export function Spinner({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-14">
      <Loader2 className="h-8 w-8 animate-spin text-[#164D3B]" />
      <p className="text-[#40534A] text-sm">{message}</p>
    </div>
  );
}

export function ProgressBar({ indeterminate = true }: { indeterminate?: boolean }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#E3EBE0]">
      <div
        className={`h-full bg-[#164D3B] ${indeterminate ? "animate-pulse w-2/3" : "w-full"} rounded-full transition-all`}
      />
    </div>
  );
}
