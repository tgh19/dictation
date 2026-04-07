import AppKit

func renderIcon(size: CGFloat) -> NSImage {
    let img = NSImage(size: NSSize(width: size, height: size))
    img.lockFocus()

    // Rounded rect gradient background
    let rect = NSRect(x: 0, y: 0, width: size, height: size)
    let radius = size * 0.22
    let path = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
    let gradient = NSGradient(
        starting: NSColor(red: 0.18, green: 0.50, blue: 0.98, alpha: 1.0),
        ending: NSColor(red: 0.08, green: 0.25, blue: 0.72, alpha: 1.0)
    )!
    gradient.draw(in: path, angle: 270)

    // Draw microphone emoji
    let emoji = "🎙️" as NSString
    let font = NSFont.systemFont(ofSize: size * 0.6)
    let attrs: [NSAttributedString.Key: Any] = [.font: font]
    let ts = emoji.size(withAttributes: attrs)
    let x = (size - ts.width) / 2
    let y = (size - ts.height) / 2
    emoji.draw(at: NSPoint(x: x, y: y), withAttributes: attrs)

    img.unlockFocus()
    return img
}

func savePNG(_ image: NSImage, to path: String, size: Int) {
    let resized = NSImage(size: NSSize(width: size, height: size))
    resized.lockFocus()
    image.draw(in: NSRect(x: 0, y: 0, width: size, height: size))
    resized.unlockFocus()

    guard let tiff = resized.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let png = rep.representation(using: .png, properties: [:]) else { return }
    try! png.write(to: URL(fileURLWithPath: path))
}

// Generate iconset
let scriptDir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().path
let iconsetDir = "\(scriptDir)/AppIcon.iconset"
try! FileManager.default.createDirectory(atPath: iconsetDir, withIntermediateDirectories: true)

let baseImage = renderIcon(size: 1024)

let entries: [(Int, String)] = [
    (16, "16x16"), (32, "16x16@2x"),
    (32, "32x32"), (64, "32x32@2x"),
    (128, "128x128"), (256, "128x128@2x"),
    (256, "256x256"), (512, "256x256@2x"),
    (512, "512x512"), (1024, "512x512@2x"),
]
for (size, suffix) in entries {
    savePNG(baseImage, to: "\(iconsetDir)/icon_\(suffix).png", size: size)
}

// Convert to icns
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconsetDir, "-o", "\(scriptDir)/AppIcon.icns"]
try! process.run()
process.waitUntilExit()

try! FileManager.default.removeItem(atPath: iconsetDir)
print("Created AppIcon.icns")
