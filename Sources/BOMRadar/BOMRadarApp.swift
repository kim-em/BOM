import SwiftUI
import AppKit
import UserNotifications

// MARK: - Configuration

private let radarID = "IDR403" // Canberra 128km
private let baseURL = "https://reg.bom.gov.au"
private let frameInterval: TimeInterval = 0.5
private let lastFramePauseTicks = 4
private let widgetSize: CGFloat = 512
private let newFrameBuffer: TimeInterval = 90
private let retryInterval: TimeInterval = 30
private let fallbackInterval: TimeInterval = 300
private let predictScript = NSString(string: "~/projects/BOM/predict.py").expandingTildeInPath

// Radar geometry: 128km range mapped to 512px image
private let radarCenterPixel: CGFloat = 256 // center of 512x512 image
private let radarRangeKm: CGFloat = 128
private let pixelsPerKm: CGFloat = 256 / 128 // 2 px/km

// MARK: - Display Mode

enum DisplayMode: String, CaseIterable {
    case past = "Past"
    case forecast = "Forecast"
    case both = "Both"
}

// MARK: - App

@main
struct BOMRadarApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    var body: some Scene {
        Settings { EmptyView() }
    }
}

// MARK: - AppDelegate

class AppDelegate: NSObject, NSApplicationDelegate, NSPopoverDelegate {
    var statusItem: NSStatusItem?
    var popover: NSPopover!
    let model = RadarModel()

    func applicationDidFinishLaunching(_: Notification) {
        NSApp.setActivationPolicy(.accessory)
        requestNotificationPermission()
        setupPopover()
        setupStatusItem()
        model.refresh()
    }

    private func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }
    }

    private func setupPopover() {
        popover = NSPopover()
        popover.contentSize = NSSize(width: widgetSize, height: widgetSize + 80)
        popover.behavior = .transient
        popover.delegate = self
        popover.contentViewController = NSHostingController(
            rootView: PopoverContentView(model: model, onQuit: {
                NSApplication.shared.terminate(nil)
            })
        )
    }

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem?.button {
            button.image = NSImage(
                systemSymbolName: "cloud.rain",
                accessibilityDescription: "BOM Radar"
            )
            button.action = #selector(togglePopover)
            button.target = self
        }
    }

    @objc func togglePopover() {
        if popover.isShown {
            popover.performClose(nil)
        } else if let button = statusItem?.button {
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            popover.contentViewController?.view.window?.makeKey()
        }
    }
}

// MARK: - Popover Content

struct PopoverContentView: View {
    @ObservedObject var model: RadarModel
    var onQuit: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            RadarView(model: model)

            // Mode picker
            Picker("", selection: $model.displayMode) {
                ForEach(DisplayMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 12)
            .padding(.top, 6)
            .onChange(of: model.displayMode) { _ in
                model.updateDisplayFrames()
            }

            // Bottom bar
            HStack {
                // Alert location indicator
                if model.alertLocation != nil {
                    HStack(spacing: 4) {
                        Image(systemName: "bell.fill")
                            .font(.caption2)
                            .foregroundColor(.orange)
                        Text("Alerts on")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Button(action: { model.alertLocation = nil }) {
                            Image(systemName: "xmark")
                                .font(.system(size: 8))
                        }
                        .buttonStyle(.borderless)
                    }
                } else {
                    Text("Click map to set rain alert")
                        .font(.caption2)
                        .foregroundColor(.secondary.opacity(0.6))
                }

                Spacer()

                if model.isPredicting {
                    ProgressView()
                        .scaleEffect(0.5)
                        .frame(width: 16, height: 16)
                }

                Button(action: { model.refresh() }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.caption)
                }
                .buttonStyle(.borderless)

                Button(action: onQuit) {
                    Image(systemName: "xmark.circle")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
    }
}

// MARK: - Radar View

struct RadarView: View {
    @ObservedObject var model: RadarModel

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 0)
                .fill(.black.opacity(0.5))

            if let img = model.currentImage {
                Image(nsImage: img)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fill)
            } else if model.isLoading {
                ProgressView()
                    .scaleEffect(1.5)
                    .tint(.white)
            } else if let error = model.errorMessage {
                VStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle)
                        .foregroundColor(.yellow)
                    Text(error)
                        .foregroundColor(.white.opacity(0.7))
                        .font(.caption)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 20)
                }
            }

            // Alert location marker
            if let loc = model.alertLocation {
                Circle()
                    .stroke(.green, lineWidth: 2)
                    .frame(width: 12, height: 12)
                    .position(x: loc.x, y: loc.y)

                Circle()
                    .fill(.green.opacity(0.3))
                    .frame(width: 12, height: 12)
                    .position(x: loc.x, y: loc.y)
            }

            // Frame progress dots — colored by past/forecast
            if model.displayFrameCount > 1 {
                VStack {
                    Spacer()
                    HStack(spacing: 3) {
                        ForEach(0 ..< model.displayFrameCount, id: \.self) { i in
                            let isForecast = i >= model.pastFrameCount && model.displayMode != .past
                            Circle()
                                .fill(i == model.currentFrameIndex
                                    ? (isForecast ? .orange : .white)
                                    : (isForecast ? .orange.opacity(0.35) : .white.opacity(0.35)))
                                .frame(width: 5, height: 5)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(.black.opacity(0.4))
                    .clipShape(Capsule())
                    .padding(.bottom, 12)
                }
            }
        }
        .frame(width: widgetSize, height: widgetSize)
        .contentShape(Rectangle())
        .onTapGesture { location in
            model.setAlertLocation(location)
        }
    }
}

// MARK: - Model

class RadarModel: ObservableObject {
    @Published var currentImage: NSImage?
    @Published var currentFrameIndex: Int = 0
    @Published var displayFrameCount: Int = 0
    @Published var pastFrameCount: Int = 0
    @Published var isLoading: Bool = false
    @Published var isPredicting: Bool = false
    @Published var errorMessage: String?
    @Published var latestFrameTime: Date?
    @Published var displayMode: DisplayMode = .both
    @Published var alertLocation: CGPoint? // pixel coords in 512x512 image

    private var pastFrames: [NSImage] = []
    private var forecastFrames: [NSImage] = []
    private var displayFrames: [NSImage] = []
    private var cachedOverlays: [NSImage]?
    private var animTimer: Timer?
    private var refreshTimer: Timer?
    private var pauseCounter = 0
    private var lastKnownFrame: String?
    private var lastNotifiedRainMinutes: Int? // avoid duplicate notifications

    func refresh() {
        let isFirstLoad = pastFrames.isEmpty
        DispatchQueue.main.async { [self] in
            if isFirstLoad { isLoading = true }
            errorMessage = nil
        }

        Task.detached { [weak self] in
            guard let self else { return }
            do {
                let overlays = try await self.fetchOverlays()
                let framePaths = try await self.fetchFramePaths()
                guard !framePaths.isEmpty else { throw RadarError.noFrames }

                let latestFrame = framePaths.last
                if latestFrame == self.lastKnownFrame && !isFirstLoad {
                    DispatchQueue.main.async { self.scheduleRefresh(after: retryInterval) }
                    return
                }

                let radarFrames = try await self.fetchImages(paths: framePaths)
                let composited = self.compositeAll(radar: radarFrames, overlays: overlays)
                let latestTime = latestFrame.flatMap { self.parseFrameTime($0) }

                DispatchQueue.main.async {
                    self.lastKnownFrame = latestFrame
                    self.latestFrameTime = latestTime
                    self.pastFrames = composited
                    self.pastFrameCount = composited.count
                    self.isLoading = false
                    self.updateDisplayFrames()
                    self.startAnimation()
                    self.scheduleNextRefresh(framePaths: framePaths)
                }

                // Run prediction in background
                self.runPrediction()

            } catch {
                DispatchQueue.main.async {
                    self.isLoading = false
                    if isFirstLoad {
                        self.errorMessage = "Could not load radar:\n\(error.localizedDescription)"
                    }
                    self.scheduleRefresh(after: fallbackInterval)
                }
            }
        }
    }

    // MARK: - Prediction

    private func runPrediction() {
        DispatchQueue.main.async { self.isPredicting = true }

        let outputDir = NSTemporaryDirectory() + "radar-predictions"
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["python3", predictScript, "--output-dir", outputDir]
        process.currentDirectoryURL = URL(fileURLWithPath: NSString(string: "~/projects/BOM").expandingTildeInPath)

        // Set up Python environment
        var env = ProcessInfo.processInfo.environment
        let venvBin = NSString(string: "~/projects/BOM/venv/bin").expandingTildeInPath
        env["PATH"] = venvBin + ":" + (env["PATH"] ?? "")
        env["VIRTUAL_ENV"] = NSString(string: "~/projects/BOM/venv").expandingTildeInPath
        process.environment = env

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
            process.waitUntilExit()

            if process.terminationStatus == 0 {
                // Load predicted frames
                var predictions: [NSImage] = []
                for i in 1...6 {
                    let path = "\(outputDir)/pred_\(i).png"
                    if let img = NSImage(contentsOfFile: path) {
                        predictions.append(img)
                    }
                }

                DispatchQueue.main.async {
                    self.forecastFrames = predictions
                    self.isPredicting = false
                    self.updateDisplayFrames()
                    self.checkRainAlerts()
                }
            } else {
                DispatchQueue.main.async { self.isPredicting = false }
            }
        } catch {
            DispatchQueue.main.async { self.isPredicting = false }
        }
    }

    // MARK: - Display Mode

    func updateDisplayFrames() {
        switch displayMode {
        case .past:
            displayFrames = pastFrames
        case .forecast:
            displayFrames = forecastFrames.isEmpty ? pastFrames : forecastFrames
        case .both:
            displayFrames = pastFrames + forecastFrames
        }

        displayFrameCount = displayFrames.count
        currentFrameIndex = 0
        if let first = displayFrames.first {
            currentImage = first
        }
        startAnimation()
    }

    // MARK: - Rain Alerts

    func setAlertLocation(_ point: CGPoint) {
        alertLocation = point
        lastNotifiedRainMinutes = nil
        checkRainAlerts()
    }

    private func checkRainAlerts() {
        guard let loc = alertLocation else { return }
        guard !forecastFrames.isEmpty else { return }

        // Convert pixel location to position in 128x128 prediction grid
        let predX = Int(loc.x / widgetSize * 128)
        let predY = Int(loc.y / widgetSize * 128)

        // Check each predicted frame for rain at this location
        // Load prediction PNGs and check pixel colors
        let outputDir = NSTemporaryDirectory() + "radar-predictions"

        for i in 1...6 {
            let path = "\(outputDir)/pred_\(i).png"
            guard let img = NSImage(contentsOfFile: path),
                  let cgImage = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }

            // Sample pixel at the alert location
            let imgX = Int(loc.x)
            let imgY = Int(loc.y)
            guard imgX >= 0, imgX < Int(img.size.width), imgY >= 0, imgY < Int(img.size.height) else { continue }

            let width = cgImage.width
            let height = cgImage.height
            let bytesPerPixel = 4
            let bytesPerRow = width * bytesPerPixel
            var pixelData = [UInt8](repeating: 0, count: bytesPerRow * height)

            guard let context = CGContext(
                data: &pixelData, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { continue }

            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

            let pixelIndex = (imgY * bytesPerRow) + (imgX * bytesPerPixel)
            let r = pixelData[pixelIndex]
            let g = pixelData[pixelIndex + 1]
            let b = pixelData[pixelIndex + 2]

            // Check if this pixel has rain (not background/black/grey)
            let isRain = detectRainLevel(r: r, g: g, b: b)
            let minutes = i * 5

            if let level = isRain, lastNotifiedRainMinutes != minutes {
                lastNotifiedRainMinutes = minutes
                sendRainNotification(level: level, minutes: minutes)
                return // only notify for first predicted rain
            }
        }
    }

    private func detectRainLevel(r: UInt8, g: UInt8, b: UInt8) -> String? {
        // Match against BOM radar colors
        if r == 245 && g == 245 && b == 255 { return "very light rain" }
        if r == 180 && g == 180 && b == 255 { return "light rain" }
        if r == 120 && g == 120 && b == 255 { return "light rain" }
        if r == 20  && g == 20  && b == 255 { return "moderate rain" }
        if r == 0   && g > 100  && b > 100  { return "moderate rain" }
        if r == 255 && g == 255 && b == 0   { return "heavy rain" }
        if r == 255 && g >= 100 && g <= 200 && b == 0 { return "heavy rain" }
        if r == 255 && g < 100  && b == 0   { return "very heavy rain" }
        if r >= 120 && r <= 200 && g == 0   && b == 0 { return "intense rain" }
        return nil
    }

    private func sendRainNotification(level: String, minutes: Int) {
        let content = UNMutableNotificationContent()
        content.title = "Rain Alert"
        content.body = "\(level.capitalized) in ~\(minutes) minutes"
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: "rain-\(minutes)",
            content: content,
            trigger: nil // deliver immediately
        )
        UNUserNotificationCenter.current().add(request)
    }

    // MARK: - Smart Refresh Scheduling

    private func parseFrameTime(_ path: String) -> Date? {
        guard let range = path.range(of: #"\d{12}"#, options: .regularExpression) else { return nil }
        let timestamp = String(path[range])
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMddHHmm"
        fmt.timeZone = TimeZone(identifier: "UTC")
        return fmt.date(from: timestamp)
    }

    private func scheduleNextRefresh(framePaths: [String]) {
        var interval: TimeInterval = 300
        if framePaths.count >= 2,
           let t1 = parseFrameTime(framePaths[framePaths.count - 2]),
           let t2 = parseFrameTime(framePaths[framePaths.count - 1])
        {
            let measured = t2.timeIntervalSince(t1)
            if measured > 0 && measured <= 600 { interval = measured }
        }

        if let lastTime = framePaths.last.flatMap({ parseFrameTime($0) }) {
            let nextCheckTime = lastTime.addingTimeInterval(interval + newFrameBuffer)
            let delay = max(10, nextCheckTime.timeIntervalSinceNow)
            scheduleRefresh(after: delay)
        } else {
            scheduleRefresh(after: fallbackInterval)
        }
    }

    private func scheduleRefresh(after delay: TimeInterval) {
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            self?.refresh()
        }
    }

    // MARK: - Animation

    private func startAnimation() {
        animTimer?.invalidate()
        guard displayFrames.count > 1 else { return }
        pauseCounter = 0

        animTimer = Timer.scheduledTimer(withTimeInterval: frameInterval, repeats: true) { [weak self] _ in
            guard let self else { return }
            if self.currentFrameIndex == self.displayFrames.count - 1 {
                self.pauseCounter += 1
                if self.pauseCounter < lastFramePauseTicks { return }
                self.pauseCounter = 0
            }
            self.currentFrameIndex = (self.currentFrameIndex + 1) % self.displayFrames.count
            self.currentImage = self.displayFrames[self.currentFrameIndex]
        }
    }

    // MARK: - Network

    private func fetchOverlays() async throws -> [NSImage] {
        if let cached = cachedOverlays { return cached }
        let names = ["background", "topography", "locations", "range"]
        var images: [NSImage] = []
        for name in names {
            guard let url = URL(string: "\(baseURL)/products/radar_transparencies/\(radarID).\(name).png") else { continue }
            let (data, resp) = try await URLSession.shared.data(from: url)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else { continue }
            if let img = NSImage(data: data) { images.append(img) }
        }
        cachedOverlays = images
        return images
    }

    private func fetchFramePaths() async throws -> [String] {
        guard let url = URL(string: "\(baseURL)/products/\(radarID).loop.shtml") else {
            throw RadarError.badURL
        }
        let (data, _) = try await URLSession.shared.data(from: url)
        let html = String(data: data, encoding: .utf8) ?? ""
        let pattern = #"/radar/IDR\d+\.T\.\d+\.png"#
        let regex = try NSRegularExpression(pattern: pattern)
        let matches = regex.matches(in: html, range: NSRange(html.startIndex..., in: html))
        return matches.compactMap { match in
            guard let range = Range(match.range, in: html) else { return nil }
            return String(html[range])
        }
    }

    private func fetchImages(paths: [String]) async throws -> [NSImage] {
        var images: [NSImage] = []
        for path in paths {
            guard let url = URL(string: "\(baseURL)\(path)") else { continue }
            let (data, resp) = try await URLSession.shared.data(from: url)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else { continue }
            if let img = NSImage(data: data) { images.append(img) }
        }
        return images
    }

    // MARK: - Compositing

    private func compositeAll(radar: [NSImage], overlays: [NSImage]) -> [NSImage] {
        return radar.map { frame in
            let size = overlays.first?.size ?? frame.size
            let w = Int(size.width)
            let h = Int(size.height)
            guard let ctx = CGContext(
                data: nil, width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return frame }
            let rect = CGRect(origin: .zero, size: size)
            for i in 0 ..< min(2, overlays.count) {
                if let cg = overlays[i].cgImage(forProposedRect: nil, context: nil, hints: nil) {
                    ctx.draw(cg, in: rect)
                }
            }
            if let cg = frame.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                ctx.draw(cg, in: rect)
            }
            for i in 2 ..< overlays.count {
                if let cg = overlays[i].cgImage(forProposedRect: nil, context: nil, hints: nil) {
                    ctx.draw(cg, in: rect)
                }
            }
            guard let result = ctx.makeImage() else { return frame }
            return NSImage(cgImage: result, size: size)
        }
    }
}

// MARK: - Errors

enum RadarError: LocalizedError {
    case badURL
    case noFrames

    var errorDescription: String? {
        switch self {
        case .badURL: return "Invalid radar URL"
        case .noFrames: return "No radar frames found"
        }
    }
}
