"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import Link from "next/link";
import { Plus, Minus } from "lucide-react";

const faqs = [
  {
    category: "About DermaSense",
    items: [
      {
        q: "What is DermaSense?",
        a: "DermaSense is an explainable multimodal AI system for skin disease detection, built specifically for Indian skin tones. It combines dermoscopic images with patient metadata to produce disease predictions, severity scores, and referral urgency — along with a Grad-CAM++ visual explanation.",
      },
      {
        q: "Is DermaSense a medical device?",
        a: "No. DermaSense is a research prototype developed as a final-year academic project at Ramdeobaba University, Nagpur. It is an AI-assisted screening aid only and does not constitute a medical diagnosis. Always consult a qualified dermatologist for clinical decisions.",
      },
      {
        q: "Why is DermaSense focused on Indian skin tones?",
        a: "Most publicly available dermatology AI systems were trained on Western datasets that predominantly feature lighter skin tones (Fitzpatrick Type I–II). Only 3.8% of standard training images represent Type IV–VI skin tones. This leads to significantly lower accuracy on Indian patients. DermaSense addresses this by training on DermaCon-IN, a dataset collected from South Indian patients across the full Fitzpatrick spectrum.",
      },
    ],
  },
  {
    category: "How It Works",
    items: [
      {
        q: "What kind of image should I upload?",
        a: "Upload a clear, well-lit photograph of the affected skin area. Dermoscopic images (taken with a dermatoscope) produce the most accurate results, but standard smartphone photos are also accepted. Ensure the lesion or area of concern is clearly visible and in focus.",
      },
      {
        q: "What patient details are required?",
        a: "You will be asked for age, gender, Fitzpatrick skin tone type, body location of the lesion, symptom duration, and any associated symptoms. All of these inputs are used by the MLP metadata encoder to improve prediction accuracy.",
      },
      {
        q: "What is a Grad-CAM++ heatmap?",
        a: "Grad-CAM++ (Gradient-weighted Class Activation Mapping++) is a technique that highlights which regions of the input image most influenced the AI model's prediction. In DermaSense, it generates a colour overlay on your image — warm colours (red, orange) indicate high influence regions, cool colours (blue, teal) indicate low influence. This makes the prediction transparent and interpretable.",
      },
      {
        q: "What does the confidence score mean?",
        a: "The confidence score represents the model's probability for its primary prediction. The remaining probability is distributed across alternative diagnoses shown in the differentials list. It is not a medical certainty.",
      },
      {
        q: "What is Cross-Attention Fusion?",
        a: "Cross-attention is a mechanism from Transformer architecture that allows the image features and metadata features to attend to each other — meaning the model can use patient context to reinterpret visual features, and vice versa. This produces a richer combined representation than simply concatenating the two feature vectors.",
      },
    ],
  },
  {
    category: "Data & Privacy",
    items: [
      {
        q: "Is my image data stored or shared?",
        a: "No. DermaSense is currently a frontend demo. Images you upload are processed entirely in your browser and are never sent to any server or stored. No patient data of any kind is retained.",
      },
      {
        q: "What datasets was DermaSense trained on?",
        a: "DermaSense uses DermaCon-IN as its primary dataset — a multi-concept annotated dermatological image dataset of Indian skin disorders, published at NeurIPS 2025 by Hasan et al. ISIC 2020 is used as a supporting dataset for additional dermoscopic lesion coverage.",
      },
    ],
  },
  {
    category: "Academic",
    items: [
      {
        q: "Who built DermaSense?",
        a: "DermaSense was built by four final-year students at Ramdeobaba University, Nagpur: Disha Kalbandhe (A-26), Hariom Rathod (A-29), Mayank Ninawe (A-39, Lead), and Mayank Bhaiya (A-40), under the supervision of Prof. Rajesh Khobragade.",
      },
      {
        q: "Can I cite DermaSense in my research?",
        a: "DermaSense is an ongoing academic project. Please contact the team through the university for citation details once the final paper is submitted.",
      },
    ],
  },
];

function FAQItem({ q, a }: { q: string; a: string }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border-b" style={{ borderColor: "var(--color-border)" }}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-start justify-between gap-4 py-5 text-left"
      >
        <span
          className="text-sm font-medium leading-snug"
          style={{ color: "var(--color-text)" }}
        >
          {q}
        </span>
        <span
          className="shrink-0 mt-0.5"
          style={{ color: "var(--color-primary)" }}
        >
          {open ? <Minus size={16} /> : <Plus size={16} />}
        </span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
            style={{ overflow: "hidden" }}
          >
            <p
              className="text-sm leading-relaxed pb-5"
              style={{ color: "var(--color-text-muted)", maxWidth: "68ch" }}
            >
              {a}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function FAQPage() {
  return (
    <main
      className="min-h-screen pt-32 pb-20 px-6"
      style={{ backgroundColor: "var(--color-bg)" }}
    >
      <div className="mx-auto" style={{ maxWidth: "var(--content-narrow)" }}>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="mb-14"
        >
          <p
            className="text-xs uppercase tracking-widest mb-3"
            style={{ color: "var(--color-text-faint)" }}
          >
            Help
          </p>
          <h1
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "var(--text-2xl)",
              color: "var(--color-text)",
              lineHeight: 1.15,
            }}
          >
            Frequently Asked Questions
          </h1>
          <p
            className="mt-4 text-base"
            style={{ color: "var(--color-text-muted)", maxWidth: "48ch" }}
          >
            Everything you need to know about DermaSense — how it works, what it
            can and cannot do, and how your data is handled.
          </p>
        </motion.div>

        {/* FAQ sections */}
        <div className="flex flex-col gap-12">
          {faqs.map((section, si) => (
            <motion.div
              key={section.category}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true, amount: 0.1 }}
              transition={{ duration: 0.4, delay: si * 0.05 }}
            >
              {/* Category label */}
              <p
                className="text-xs uppercase tracking-widest mb-2"
                style={{ color: "var(--color-text-faint)" }}
              >
                {section.category}
              </p>

              {/* Items */}
              <div
                className="rounded-2xl overflow-hidden px-5"
                style={{
                  backgroundColor: "var(--color-surface-2)",
                  border: "1px solid var(--color-border)",
                  boxShadow: "var(--shadow-sm)",
                }}
              >
                {section.items.map((item) => (
                  <FAQItem key={item.q} q={item.q} a={item.a} />
                ))}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Bottom CTA */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="mt-14 p-6 rounded-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
          style={{
            backgroundColor: "var(--color-surface)",
            border: "1px solid var(--color-border)",
          }}
        >
          <div>
            <p
              className="text-sm font-medium mb-1"
              style={{ color: "var(--color-text)" }}
            >
              Ready to try it?
            </p>
            <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
              Upload a skin image and get an AI-powered analysis in seconds.
            </p>
          </div>
          <Link
            href="/analyze"
            className="px-6 py-3 rounded-full text-sm font-medium shrink-0"
            style={{
              backgroundColor: "var(--color-primary)",
              color: "#ffffff",
            }}
          >
            Analyze Skin →
          </Link>
        </motion.div>
      </div>
    </main>
  );
}
