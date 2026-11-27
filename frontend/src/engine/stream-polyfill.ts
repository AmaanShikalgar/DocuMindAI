/** WebKit (Safari, and every browser on iOS) can't `for await` over a ReadableStream yet, and
 * pdf.js's getTextContent does exactly that. Import this before pdf.js. */
type IterableStream = ReadableStream & { [Symbol.asyncIterator]?: unknown; values?: unknown };
const proto = ReadableStream.prototype as IterableStream;

if (typeof proto[Symbol.asyncIterator] !== "function") {
  const values = async function* (this: ReadableStream, opts?: { preventCancel?: boolean }) {
    const reader = this.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) return;
        yield value;
      }
    } finally {
      if (!opts?.preventCancel) await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  };
  Object.defineProperty(proto, "values", { value: values, writable: true, configurable: true });
  Object.defineProperty(proto, Symbol.asyncIterator, { value: values, writable: true, configurable: true });
}
