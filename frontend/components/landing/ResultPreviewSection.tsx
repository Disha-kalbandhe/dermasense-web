"use client";
import { useRef, useState } from "react";
import {
  motion,
  useInView,
  useMotionValue,
  useSpring,
  useTransform,
} from "motion/react";

const differentials = [
  { condition: "Example condition", confidence: 42 },
  { condition: "Illustrative alternative", confidence: 24 },
  { condition: "Illustrative possibility", confidence: 12 },
];

function ConfidenceBar({
  confidence,
  index,
}: {
  confidence: number;
  index: number;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  return (
    <div ref={ref} className="flex items-center gap-3">
      <div
        className="flex-1 h-1.5 rounded-full overflow-hidden"
        style={{ backgroundColor: "var(--color-surface-offset)" }}
      >
        <motion.div
          className="h-full rounded-full"
          style={{
            backgroundColor:
              index === 0 ? "var(--color-primary)" : "var(--color-text-faint)",
          }}
          initial={{ width: 0 }}
          animate={inView ? { width: `${confidence}%` } : {}}
          transition={{
            duration: 0.8,
            delay: index * 0.1 + 0.3,
            ease: [0.16, 1, 0.3, 1],
          }}
        />
      </div>
      <span
        className="text-xs shrink-0 w-10 text-right"
        style={{
          fontFamily: "var(--font-mono)",
          color: "var(--color-text-muted)",
        }}
      >
        {confidence}%
      </span>
    </div>
  );
}

export default function ResultPreviewSection() {
  const sectionRef = useRef(null);
  const inView = useInView(sectionRef, { once: true, margin: "-80px" });
  const [activeTab, setActiveTab] = useState<"original" | "heatmap">("heatmap");

  // Mouse parallax tilt — desktop only
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const rotateX = useSpring(useTransform(y, [-0.5, 0.5], [6, -6]), {
    stiffness: 200,
    damping: 30,
  });
  const rotateY = useSpring(useTransform(x, [-0.5, 0.5], [-6, 6]), {
    stiffness: 200,
    damping: 30,
  });

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    x.set((e.clientX - rect.left) / rect.width - 0.5);
    y.set((e.clientY - rect.top) / rect.height - 0.5);
  }
  function handleMouseLeave() {
    x.set(0);
    y.set(0);
  }

  return (
    <section
      ref={sectionRef}
      className="py-24 px-6 pb-32"
      style={{ backgroundColor: "var(--color-bg)" }}
    >
      <div
        className="mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-center"
        style={{ maxWidth: "var(--content-wide)" }}
      >
        {/* Left — text */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.6 }}
        >
          <p
            className="text-xs uppercase tracking-widest mb-4"
            style={{ color: "var(--color-text-faint)" }}
          >
            Sample Output
          </p>
          <h2
            className="mb-4"
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "var(--text-xl)",
              color: "var(--color-text)",
              lineHeight: 1.2,
            }}
          >
            Every prediction comes with a visual explanation
          </h2>
          <p
            className="text-base mb-8 leading-relaxed"
            style={{ color: "var(--color-text-muted)", maxWidth: "48ch" }}
          >
            The Grad-CAM++ heatmap highlights exactly which region of the skin
            influenced the prediction — so patients and clinicians understand
            the reasoning, not just the result.
          </p>
          <a
            href="/analyze"
            className="inline-block text-sm font-medium px-6 py-3 rounded-full transition-opacity hover:opacity-90"
            style={{
              backgroundColor: "var(--color-primary)",
              color: "#ffffff",
            }}
          >
            Try with your own image →
          </a>
        </motion.div>

        {/* Right — floating result card */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.7, delay: 0.2 }}
          style={{ perspective: 1000 }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          <motion.div
            style={{
              rotateX,
              rotateY,
              transformStyle: "preserve-3d",
            }}
            className="rounded-2xl overflow-hidden"
          >
            {/* Card */}
            <div
              className="rounded-2xl overflow-hidden"
              style={{
                backgroundColor: "var(--color-surface-2)",
                boxShadow: "var(--shadow-lg)",
                border: "1px solid var(--color-border)",
              }}
            >
              {/* Image area with tab toggle */}
              <div className="relative">
                {/* Tab switcher */}
                <div
                  className="absolute top-3 right-3 z-10 flex rounded-full overflow-hidden text-xs"
                  style={{
                    backgroundColor: "rgba(0,0,0,0.45)",
                    backdropFilter: "blur(6px)",
                  }}
                >
                  {(["original", "heatmap"] as const).map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setActiveTab(tab)}
                      className="px-3 py-1.5 capitalize transition-colors"
                      style={{
                        color:
                          activeTab === tab
                            ? "#ffffff"
                            : "rgba(255,255,255,0.55)",
                        backgroundColor:
                          activeTab === tab
                            ? "var(--color-primary)"
                            : "transparent",
                      }}
                    >
                      {tab === "heatmap" ? "Heatmap" : "Original"}
                    </button>
                  ))}
                </div>

                {/* Illustrative preview; live explanations appear on the analysis result page. */}
                <div
                  className="w-full h-52 flex items-center justify-center relative overflow-hidden"
                  style={{
                    backgroundColor:
                      activeTab === "heatmap"
                        ? "#1a1a2e"
                        : "var(--color-surface-offset)",
                  }}
                >
                  {activeTab === "heatmap" ? (
                    <div className="absolute inset-0">
                      <div
                        className="absolute inset-0"
                        style={{
                          background:
                            "radial-gradient(ellipse 60% 50% at 45% 55%, #ff4500 0%, #ff8c00 30%, #ffd700 55%, #00ced1 80%, #1a1a2e 100%)",
                          opacity: 0.85,
                        }}
                      />
                      <div
                        className="absolute inset-0 flex items-center justify-center"
                        style={{
                          color: "rgba(255,255,255,0.3)",
                          fontSize: "11px",
                          fontFamily: "var(--font-mono)",
                        }}
                      >
                        ILLUSTRATIVE PREVIEW · NOT MODEL OUTPUT
                      </div>
                    </div>
                  ) : (
                    <div
                      className="w-32 h-32 rounded-full"
                      style={{
                        backgroundColor:
                          "var(--color-surface-dynamic, #e0ddd7)",
                        border: "1px solid var(--color-border)",
                      }}
                    />
                  )}
                </div>
              </div>

              {/* Result details */}
              <div className="p-5">
                {/* Condition */}
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p
                      className="text-xs uppercase tracking-widest mb-1"
                      style={{ color: "var(--color-text-faint)" }}
                    >
                      Predicted Condition
                    </p>
                    <p
                      className="text-base font-semibold"
                      style={{ color: "var(--color-text)" }}
                    >
                      Example condition
                    </p>
                  </div>
                </div>

                {/* Confidence */}
                <div className="mb-4">
                  <div className="flex justify-between items-center mb-1.5">
                    <p
                      className="text-xs uppercase tracking-widest"
                      style={{ color: "var(--color-text-faint)" }}
                    >
                      Confidence
                    </p>
                    <span
                      className="text-sm font-medium"
                      style={{
                        fontFamily: "var(--font-mono)",
                        color: "var(--color-primary)",
                      }}
                    >
                      Example only
                    </span>
                  </div>
                  <div
                    className="h-1.5 rounded-full overflow-hidden"
                    style={{ backgroundColor: "var(--color-surface-offset)" }}
                  >
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: "var(--color-primary)" }}
                      initial={{ width: 0 }}
                      animate={inView ? { width: "42%" } : {}}
                      transition={{
                        duration: 1,
                        delay: 0.5,
                        ease: [0.16, 1, 0.3, 1],
                      }}
                    />
                  </div>
                </div>

                {/* Differentials */}
                <div className="mb-4">
                  <p
                    className="text-xs uppercase tracking-widest mb-3"
                    style={{ color: "var(--color-text-faint)" }}
                  >
                    Top 3 Differentials
                  </p>
                  <div className="flex flex-col gap-2.5">
                    {differentials.map((d, i) => (
                      <div key={d.condition}>
                        <div className="flex justify-between items-center mb-1">
                          <span
                            className="text-xs"
                            style={{
                              color:
                                i === 0
                                  ? "var(--color-text)"
                                  : "var(--color-text-muted)",
                            }}
                          >
                            {d.condition}
                          </span>
                        </div>
                        <ConfidenceBar confidence={d.confidence} index={i} />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Referral */}
                <div
                  className="flex items-center gap-2 pt-4 border-t"
                  style={{ borderColor: "var(--color-border)" }}
                >
                  <div
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: "var(--color-mild)" }}
                  />
                  <p
                    className="text-xs"
                    style={{ color: "var(--color-text-muted)" }}
                  >
                    No clinical recommendation is generated by this preview.
                  </p>
                </div>

                {/* Disclaimer */}
                <div
                  className="mt-3 p-3 rounded-lg text-xs leading-relaxed"
                  style={{
                    backgroundColor: "var(--color-saffron-light)",
                    color: "var(--color-saffron)",
                  }}
                >
                  ⚠ AI screening aid only. Consult a qualified dermatologist
                  before any action.
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
