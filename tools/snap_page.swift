// Full-page screenshot of a URL with macOS WebKit (WKWebView): for checking the app's page
// where headless Brave fails (JOURNAL Phase 7). No browser download needed.
//
//   swiftc -O tools/snap_page.swift -o /tmp/snap_page
//   /tmp/snap_page <url> <out.png> <width> <light|dark> [js-to-run-after-load] [wait-s]
//   /tmp/snap_page http://127.0.0.1:8000/ /tmp/example.png 1280 dark \
//       "document.getElementById('example-btn').click()" 2
//
// The window sits offscreen; the PNG is at the screen's scale (2x on a Retina Mac).
import AppKit
import WebKit

let args = CommandLine.arguments
let url = URL(string: args[1])!
let out = args[2]
let width = CGFloat(Double(args[3]) ?? 1280)
let dark = args.count > 4 && args[4] == "dark"
let js = args.count > 5 ? args[5] : ""
let wait = args.count > 6 ? Double(args[6]) ?? 1.5 : 1.5

let app = NSApplication.shared
app.setActivationPolicy(.prohibited)

final class Snapper: NSObject, WKNavigationDelegate {
    let view: WKWebView
    let window: NSWindow
    init(view: WKWebView, window: NSWindow) { self.view = view; self.window = window }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            webView.evaluateJavaScript(js.isEmpty ? "0" : js) { _, err in
                if let err = err { FileHandle.standardError.write("js error: \(err)\n".data(using: .utf8)!) }
                DispatchQueue.main.asyncAfter(deadline: .now() + wait) { self.capture() }
            }
        }
    }

    func capture() {
        view.evaluateJavaScript("document.documentElement.scrollHeight") { h, _ in
            let height = CGFloat((h as? NSNumber)?.doubleValue ?? 1200)
            self.window.setContentSize(NSSize(width: width, height: height))
            self.view.frame = NSRect(x: 0, y: 0, width: width, height: height)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                let cfg = WKSnapshotConfiguration()
                cfg.rect = NSRect(x: 0, y: 0, width: width, height: height)
                self.view.takeSnapshot(with: cfg) { image, err in
                    guard let image = image, let tiff = image.tiffRepresentation,
                          let rep = NSBitmapImageRep(data: tiff),
                          let png = rep.representation(using: .png, properties: [:]) else {
                        FileHandle.standardError.write("snapshot failed: \(String(describing: err))\n".data(using: .utf8)!)
                        exit(1)
                    }
                    try! png.write(to: URL(fileURLWithPath: out))
                    print("wrote \(out) \(Int(width))x\(Int(height))")
                    exit(0)
                }
            }
        }
    }
}

let window = NSWindow(contentRect: NSRect(x: -20000, y: -20000, width: width, height: 1000),
                      styleMask: [.borderless], backing: .buffered, defer: false)
window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
let view = WKWebView(frame: NSRect(x: 0, y: 0, width: width, height: 1000))
window.contentView = view
window.orderBack(nil)
let snapper = Snapper(view: view, window: window)
view.navigationDelegate = snapper
view.load(URLRequest(url: url))
DispatchQueue.main.asyncAfter(deadline: .now() + 60) { print("timeout"); exit(2) }
app.run()
