interface Props {
  label: "REAL" | "FAKE";
  confidence: number;
  fakeProbability: number;
  realProbability: number;
}

export default function ConfidenceBar({ label, confidence, fakeProbability, realProbability }: Props) {
  const isFake = label === "FAKE";
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2">
        <span
          className={`px-3 py-1 rounded-full text-sm font-bold tracking-wide ${
            isFake ? "bg-[#B94B45]/20 text-[#B94B45] border border-[#E3B5B0]" : "bg-[#28734D]/20 text-[#28734D] border border-[#9DC8AA]"
          }`}
        >
          {label}
        </span>
        <span className="text-[#40534A] text-sm">{(confidence * 100).toFixed(1)}% confidence</span>
      </div>

      <div className="flex h-3 w-full rounded-full overflow-hidden bg-[#E3EBE0]">
        <div className="bg-[#28734D] transition-all duration-700" style={{ width: `${realProbability * 100}%` }} />
        <div className="bg-[#B94B45] transition-all duration-700" style={{ width: `${fakeProbability * 100}%` }} />
      </div>
      <div className="flex justify-between text-xs text-[#5F6868] mt-1">
        <span>Real: {(realProbability * 100).toFixed(1)}%</span>
        <span>Fake: {(fakeProbability * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}
