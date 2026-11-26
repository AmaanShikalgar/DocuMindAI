/** pdf.js set up for the browser (worker bundled by Vite), plus a small cache of open documents
 * for the page viewer. */
import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

export { pdfjs };

const open = new Map<string, Promise<pdfjs.PDFDocumentProxy>>();

/** Open (once) a PDF from a URL or bytes. Bytes are copied, because pdf.js takes ownership. */
export function getPdf(key: string, source: string | Uint8Array): Promise<pdfjs.PDFDocumentProxy> {
  if (!open.has(key)) {
    const params = typeof source === "string" ? { url: source } : { data: source.slice() };
    open.set(key, pdfjs.getDocument(params).promise);
  }
  return open.get(key)!;
}

export function forgetPdf(key: string) {
  open.get(key)?.then((d) => d.destroy());
  open.delete(key);
}
