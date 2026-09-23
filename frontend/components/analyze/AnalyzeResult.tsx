"use client";
import { motion } from "motion/react";
import type { FormData } from "@/components/analyze/types";
import type { PredictResult } from "@/lib/api";

function getConfidenceLabel(confidence: number) {
  if (confidence >= 0.7) return "HIGH";
  if (confidence >= 0.4) return "MODERATE";
  return "LOW";
}

function getConfidenceBadgeStyle(confidence: number) {
  if (confidence >= 0.7) {
    return { backgroundColor: "#dcfce7", color: "#166534" };
  }
  if (confidence >= 0.4) {
    return { backgroundColor: "#fef3c7", color: "#92400e" };
  }
  return { backgroundColor: "#fee2e2", color: "#991b1b" };
}

export default function AnalyzeResult({
  formData,
  result,
  onReset,
}: {
  formData: FormData;
  result: PredictResult;
  onReset: () => void;
}) {
  const differentials = result.differentials;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-10">
        <div>
          <p
            className="text-xs uppercase tracking-widest mb-2"
            style={{ color: "var(--color-text-faint)" }}
          >
            Analysis Complete
          </p>
          <h1
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "var(--text-xl)",
              color: "var(--color-text)",
            }}
          >
            Results
          </h1>
        </div>
        <button
          onClick={onReset}
          className="text-sm px-5 py-2.5 rounded-full transition-colors"
          style={{
            border: "1px solid var(--color-border)",
            color: "var(--color-text-muted)",
            backgroundColor: "var(--color-surface)",
          }}
        >
          ← New Analysis
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
        {/* Left — Heatmap */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <p
              className="text-xs uppercase tracking-widest"
              style={{ color: "var(--color-text-faint)" }}
            >
              Grad-CAM++ Visualization
            </p>
            <span
              className="text-xs"
              style={{ color: "var(--color-text-faint)" }}
            >
              {result.explainability.available
                ? "Model-derived"
                : "Unavailable"}
            </span>
          </div>

          {/* Image area */}
          <div
            className="w-full aspect-square rounded-2xl overflow-hidden flex items-center justify-center relative"
            style={{ backgroundColor: "var(--color-surface-offset)" }}
          >
            {result.explainability.available &&
            result.explainability.heatmap_base64 ? (
              <img
                src={result.explainability.heatmap_base64}
                alt="Grad-CAM++ model explanation"
                className="w-full h-full object-cover"
              />
            ) : (
              <p
                className="text-sm"
                style={{ color: "var(--color-text-muted)" }}
              >
                Explanation unavailable for this result.
              </p>
            )}
          </div>

          {/* Patient summary */}
          <div
            className="mt-4 p-4 rounded-xl grid grid-cols-3 gap-3"
            style={{
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
            }}
          >
            {[
              { label: "Age", value: `${formData.age} yrs` },
              { label: "Skin Tone", value: `Type ${formData.skinTone}` },
              { label: "Location", value: formData.bodyLocation },
              { label: "Gender", value: formData.gender },
              {
                label: "Duration",
                value: formData.duration.split(" ").slice(0, 2).join(" "),
              },
              {
                label: "Symptoms",
                value:
                  formData.symptoms.length > 0
                    ? formData.symptoms.slice(0, 2).join(", ")
                    : "None",
              },
            ].map((item) => (
              <div key={item.label}>
                <p
                  className="text-xs uppercase tracking-widest mb-0.5"
                  style={{ color: "var(--color-text-faint)" }}
                >
                  {item.label}
                </p>
                <p
                  className="text-sm font-medium"
                  style={{ color: "var(--color-text)" }}
                >
                  {item.value}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Right — Result card */}
        <div className="flex flex-col gap-5">
          {/* Primary diagnosis */}
          <div
            className="p-6 rounded-2xl"
            style={{
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
            }}
          >
            <div className="flex items-start justify-between mb-1">
              <p
                className="text-xs uppercase tracking-widest"
                style={{ color: "var(--color-text-faint)" }}
              >
                Primary Diagnosis
              </p>
              <span
                className="text-xs font-medium px-2.5 py-1 rounded-full"
                style={getConfidenceBadgeStyle(result.primary_confidence)}
              >
                {getConfidenceLabel(result.primary_confidence)}
              </span>
            </div>
            <h2
              className="mt-2 mb-4"
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "var(--text-lg)",
                color: "var(--color-text)",
              }}
            >
              {result.primary_condition}
            </h2>
            {/* Confidence bar */}
            <div className="flex justify-between items-center mb-1.5">
              <p
                className="text-xs"
                style={{ color: "var(--color-text-faint)" }}
              >
                Confidence
              </p>
              <span
                className="text-sm font-medium"
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  color: "var(--color-primary)",
                }}
              >
                {(result.primary_confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div
              className="h-2 rounded-full overflow-hidden"
              style={{ backgroundColor: "var(--color-surface-offset)" }}
            >
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: "var(--color-primary)" }}
                initial={{ width: 0 }}
                animate={{
                  width: `${Math.min(result.primary_confidence * 100, 100)}%`,
                }}
                transition={{
                  duration: 1,
                  delay: 0.3,
                  ease: [0.16, 1, 0.3, 1],
                }}
              />
            </div>
          </div>

          {/* Differentials */}
          <div
            className="p-6 rounded-2xl"
            style={{
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
            }}
          >
            <p
              className="text-xs uppercase tracking-widest mb-4"
              style={{ color: "var(--color-text-faint)" }}
            >
              Top 3 Differentials
            </p>
            <div className="flex flex-col gap-4">
              {differentials.map((d, i) => (
                <div key={d.condition}>
                  <div className="flex justify-between items-center mb-1.5">
                    <span
                      className="text-sm"
                      style={{
                        color:
                          i === 0
                            ? "var(--color-text)"
                            : "var(--color-text-muted)",
                        fontWeight: i === 0 ? 500 : 400,
                      }}
                    >
                      {d.condition}
                    </span>
                    <span
                      className="text-xs"
                      style={{
                        fontFamily: "var(--font-mono, monospace)",
                        color: "var(--color-text-muted)",
                      }}
                    >
                      {(d.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div
                    className="h-1.5 rounded-full overflow-hidden"
                    style={{ backgroundColor: "var(--color-surface-offset)" }}
                  >
                    <motion.div
                      className="h-full rounded-full"
                      style={{
                        backgroundColor:
                          i === 0
                            ? "var(--color-primary)"
                            : "var(--color-text-faint)",
                      }}
                      initial={{ width: 0 }}
                      animate={{
                        width: `${Math.min(d.confidence * 100, 100)}%`,
                      }}
                      transition={{
                        duration: 0.8,
                        delay: 0.4 + i * 0.1,
                        ease: [0.16, 1, 0.3, 1],
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Hierarchy */}
          <div
            className="p-5 rounded-2xl"
            style={{
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
            }}
          >
            <p
              className="text-xs uppercase tracking-widest mb-3"
              style={{ color: "var(--color-text-faint)" }}
            >
              Predicted hierarchy
            </p>
            <p className="text-sm" style={{ color: "var(--color-text)" }}>
              {result.hierarchy_path.join(" / ")}
            </p>
            <p
              className="text-xs mt-2"
              style={{ color: "var(--color-text-muted)" }}
            >
              Mainclass{" "}
              {(result.hierarchy_confidence.mainclass * 100).toFixed(1)}% ·
              Subclass {(result.hierarchy_confidence.subclass * 100).toFixed(1)}
              % ·{" "}
              {result.hierarchical_consistency
                ? "Hierarchy consistent"
                : "Hierarchy heads disagree"}
            </p>
          </div>

          {/* Disclaimer */}
          <div
            className="p-4 rounded-xl text-xs leading-relaxed"
            style={{
              backgroundColor: "var(--color-accent-light, #FDECD6)",
              color: "var(--color-accent, #D4700A)",
            }}
          >
            ⚠ This is an AI-assisted screening tool only. These results do not
            constitute a medical diagnosis. Please consult a qualified
            dermatologist before taking any action.
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={onReset}
              className="flex-1 py-3 rounded-full text-sm font-medium transition-colors"
              style={{
                backgroundColor: "var(--color-primary)",
                color: "#ffffff",
              }}
            >
              Analyze Another →
            </button>
            <button
              className="px-5 py-3 rounded-full text-sm transition-colors"
              style={{
                border: "1px solid var(--color-border)",
                color: "var(--color-text-muted)",
                backgroundColor: "var(--color-surface)",
              }}
              onClick={() => window.print()}
            >
              Save Report
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
