import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

/** Inline or display TeX, rendered with KaTeX. */
export function TeX({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: block, throwOnError: false, strict: false }),
    [children, block],
  );
  return block
    ? <div className="math-block" dangerouslySetInnerHTML={{ __html: html }} />
    : <span dangerouslySetInnerHTML={{ __html: html }} />;
}
