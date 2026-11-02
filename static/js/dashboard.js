/**
 * CLASSVISION PRO: Executive Institutional Intelligence
 * Client Engine: Apple Audio Synthesizer, Acoustic Snap Detector,
 * MediaPipe Air-Drawing Canvas, and Telemetry Engine.
 */

// ==========================================================================
// 1. APPLE AUDIO SYNTHESIZER (Refined VisionOS / macOS Soundscapes)
// ==========================================================================
class AppleAudio {
    constructor() {
        this.ctx = null;
        this.enabled = true;
    }

    init() {
        if (!this.ctx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AudioContext();
        }
    }

    playCrystalActivate() {
        if (!this.enabled) return;
        this.init();
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const now = this.ctx.currentTime;
        // Two pure sine bell tones: E6 (1318 Hz) -> B6 (1975 Hz)
        [1318.5, 1975.5].forEach((freq, idx) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now + idx * 0.08);

            gain.gain.setValueAtTime(0.08, now + idx * 0.08);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + idx * 0.08 + 0.35);

            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start(now + idx * 0.08);
            osc.stop(now + idx * 0.08 + 0.35);
        });
    }

    playWipeClear() {
        if (!this.enabled) return;
        this.init();
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const now = this.ctx.currentTime;
        // Smooth descending sweep
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, now);
        osc.frequency.exponentialRampToValueAtTime(220, now + 0.25);

        gain.gain.setValueAtTime(0.07, now);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.25);

        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(now);
        osc.stop(now + 0.25);
    }

    playAlertChime() {
        if (!this.enabled) return;
        this.init();
        if (this.ctx.state === 'suspended') this.ctx.resume();

        const now = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(587.33, now); // D5
        osc.frequency.setValueAtTime(440.0, now + 0.12); // A4

        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(now);
        osc.stop(now + 0.35);
    }
}

// ==========================================================================
// 2. ACOUSTIC FINGER SNAP DETECTOR (Web Audio API Transient Analyser)
// ==========================================================================
class SnapDetector {
    constructor(onSnapCallback) {
        this.onSnap = onSnapCallback;
        this.audioCtx = null;
        this.analyser = null;
        this.micStream = null;
        this.isListening = false;
        this.lastSnapTime = 0;
        this.prevEnergy = 0;
    }

    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            this.micStream = stream;
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioCtx = new AudioContext();
            const source = this.audioCtx.createMediaStreamSource(stream);

            this.analyser = this.audioCtx.createAnalyser();
            this.analyser.fftSize = 512;
            this.analyser.smoothingTimeConstant = 0.1; // Fast transient response
            source.connect(this.analyser);

            this.isListening = true;
            this.detectLoop();
            console.log("[SnapDetector] Microphone armed. Listening for acoustic finger snaps...");
            return true;
        } catch (err) {
            console.warn("[SnapDetector] Microphone permission not granted:", err);
            return false;
        }
    }

    detectLoop() {
        if (!this.isListening) return;

        const buffer = new Uint8Array(this.analyser.frequencyBinCount);
        this.analyser.getByteFrequencyData(buffer);

        // A finger snap concentrates sharp high-frequency energy in 2.5kHz - 6kHz (bins 30 to 75)
        let highFreqEnergy = 0;
        for (let i = 28; i < 80; i++) {
            highFreqEnergy += buffer[i];
        }
        highFreqEnergy /= 52;

        // Human speech vocal band (300Hz - 2400Hz: bins 4 to 26)
        let vocalEnergy = 0;
        for (let i = 4; i < 26; i++) {
            vocalEnergy += buffer[i];
        }
        vocalEnergy /= 22;

        // Acoustic finger snap concentrates high-frequency transient energy
        const now = Date.now();
        const delta = highFreqEnergy - this.prevEnergy;

        if (delta > 35 && highFreqEnergy > 60 && (now - this.lastSnapTime > 750)) {
            this.lastSnapTime = now;
            console.log("⚡ [SnapDetector] Acoustic Finger Snap Detected!");
            if (this.onSnap) this.onSnap();
        }

        this.prevEnergy = highFreqEnergy * 0.7; // Smooth decay
        requestAnimationFrame(() => this.detectLoop());
    }

    stop() {
        this.isListening = false;
        if (this.micStream) {
            this.micStream.getTracks().forEach(t => t.stop());
        }
    }
}

// ==========================================================================
// 3. AIR-DRAWING CANVAS & PINCH GESTURE ENGINE
// ==========================================================================
class AirDrawEngine {
    constructor(canvasEl, cursorEl, paletteEl, onStateChange) {
        this.canvas = canvasEl;
        this.ctx = canvasEl.getContext('2d');
        this.cursor = cursorEl;
        this.palette = paletteEl;
        this.onStateChange = onStateChange;

        this.active = false;
        this.isDrawing = false;
        this.currentColor = '#0A84FF'; // Apple System Blue
        this.lineWidth = 4;
        this.lastX = 0;
        this.lastY = 0;

        // MediaPipe Hands instance
        this.mpHands = null;
        this.camera = null;

        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.initMouseDrawing();
        this.initPalette();
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    toggle() {
        this.active = !this.active;
        if (this.palette) this.palette.classList.toggle('active', this.active);

        if (this.active) {
            if (this.cursor) this.cursor.style.display = 'block';
            if (this.onStateChange) this.onStateChange(true);
            this.initMediaPipe();
        } else {
            this.clear();
            if (this.cursor) this.cursor.style.display = 'none';
            if (this.onStateChange) this.onStateChange(false);
        }
    }

    initPalette() {
        if (!this.palette) return;
        // Color selection
        this.palette.querySelectorAll('.color-dot').forEach(btn => {
            btn.addEventListener('click', () => {
                this.palette.querySelectorAll('.color-dot').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentColor = btn.getAttribute('data-color');
            });
        });

        const clearBtn = document.getElementById('btn-palette-clear');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clear();
            });
        }

        const exitBtn = document.getElementById('btn-palette-exit');
        if (exitBtn) {
            exitBtn.addEventListener('click', () => {
                this.toggle();
            });
        }
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    // Touch / Mouse drawing support as seamless fallback
    initMouseDrawing() {
        this.canvas.addEventListener('mousedown', (e) => {
            if (!this.active) return;
            this.isDrawing = true;
            const rect = this.canvas.getBoundingClientRect();
            this.lastX = e.clientX - rect.left;
            this.lastY = e.clientY - rect.top;
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.active) return;
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Move cursor
            this.cursor.style.left = `${x}px`;
            this.cursor.style.top = `${y}px`;

            if (this.isDrawing) {
                this.drawSegment(this.lastX, this.lastY, x, y);
                this.lastX = x;
                this.lastY = y;
            }
        });

        window.addEventListener('mouseup', () => {
            this.isDrawing = false;
        });
    }

    drawSegment(x1, y1, x2, y2) {
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.strokeStyle = this.currentColor;
        this.ctx.lineWidth = this.lineWidth;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.shadowColor = this.currentColor;
        this.ctx.shadowBlur = 8;
        this.ctx.stroke();
    }

    // Initialize MediaPipe Hands for Real-Time Finger Pinch Tracking
    initMediaPipe() {
        if (this.mpHands || typeof window.Hands === 'undefined') return;

        try {
            this.mpHands = new window.Hands({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
            });

            this.mpHands.setOptions({
                maxNumHands: 1,
                modelComplexity: 0, // 0 = ultra-fast real-time
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            this.mpHands.onResults((results) => this.onHandResults(results));

            // Hook into video feed if available
            const streamImg = document.getElementById('live-stream-img');
            // Poll frames for hand landmarks
            const processVideoFrame = async () => {
                if (this.active && this.mpHands && streamImg.complete && streamImg.naturalWidth > 0) {
                    try {
                        await this.mpHands.send({ image: streamImg });
                    } catch (e) {}
                }
                if (this.active) {
                    requestAnimationFrame(processVideoFrame);
                }
            };
            requestAnimationFrame(processVideoFrame);

        } catch (e) {
            console.log("[AirDraw] MediaPipe Hands running in fallback mode:", e);
        }
    }

    onHandResults(results) {
        if (!this.active || !results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
            this.isDrawing = false;
            this.cursor.classList.remove('drawing');
            return;
        }

        const landmarks = results.multiHandLandmarks[0];
        // 4: Thumb Tip, 8: Index Tip
        const thumb = landmarks[4];
        const index = landmarks[8];

        const w = this.canvas.width;
        const h = this.canvas.height;

        // Coordinates on canvas
        const pinchX = ((thumb.x + index.x) / 2.0) * w;
        const pinchY = ((thumb.y + index.y) / 2.0) * h;

        // Euclidean distance in normalized coords
        const dist = Math.hypot(thumb.x - index.x, thumb.y - index.y);

        // Position Apple Vision target cursor
        this.cursor.style.left = `${pinchX}px`;
        this.cursor.style.top = `${pinchY}px`;

        // If distance < 0.055: Thumb and Index are touching/pinched!
        if (dist < 0.055) {
            this.cursor.classList.add('drawing');
            document.getElementById('palette-status').textContent = 'Pinch Active • Drawing';
            if (this.isDrawing) {
                this.drawSegment(this.lastX, this.lastY, pinchX, pinchY);
            }
            this.isDrawing = true;
            this.lastX = pinchX;
            this.lastY = pinchY;
        } else {
            this.cursor.classList.remove('drawing');
            document.getElementById('palette-status').textContent = 'Pinch Thumb & Index to Draw';
            this.isDrawing = false;
        }
    }
}

// ==========================================================================
// 4. MAIN DASHBOARD APPLICATION STATE & CONTROLLER
// ==========================================================================
const appleAudio = new AppleAudio();
let airDrawEngine = null;
let snapDetector = null;
let engagementChart = null;
let lastIncidentId = null;
let isHeatmapActive = false;

document.addEventListener('DOMContentLoaded', () => {
    initClock();
    initChart();
    initAirDrawingAndSnap();
    initControls();
    startPollingTelemetry();
    startPollingIncidents();

    // Enable audio context on first user click
    document.addEventListener('click', () => {
        appleAudio.init();
    }, { once: true });
});

// Clock
function initClock() {
    const clockEl = document.getElementById('digital-clock');
    setInterval(() => {
        const now = new Date();
        clockEl.textContent = now.toTimeString().split(' ')[0];
    }, 1000);
}

// Air Draw and Snap Detector setup
function initAirDrawingAndSnap() {
    const canvas = document.getElementById('drawing-canvas');
    const cursor = document.getElementById('finger-cursor');
    const palette = document.getElementById('air-draw-palette');
    const dynamicIsland = document.getElementById('dynamic-island');
    const airBtn = document.getElementById('btn-air-draw');

    airDrawEngine = new AirDrawEngine(canvas, cursor, palette, (isActive) => {
        if (isActive) {
            dynamicIsland.classList.add('expanded');
            airBtn.classList.remove('nier-btn-secondary');
            airBtn.classList.add('nier-btn-primary');
            appleAudio.playCrystalActivate();
        } else {
            dynamicIsland.classList.remove('expanded');
            airBtn.classList.remove('nier-btn-primary');
            airBtn.classList.add('nier-btn-secondary');
            appleAudio.playWipeClear();
        }
    });

    // Acoustic Snap Detection callback
    snapDetector = new SnapDetector(() => {
        // Toggle drawing on snap! If currently active, snap removes/clears the drawing!
        airDrawEngine.toggle();
    });

    // Microphone toggle button
    const micBtn = document.getElementById('btn-mic-toggle');
    micBtn.addEventListener('click', async () => {
        if (!snapDetector.isListening) {
            const ok = await snapDetector.start();
            if (ok) {
                document.getElementById('mic-icon').className = 'fa-solid fa-microphone text-mint';
                showDynamicIslandMessage("Snap Detector Armed • Snap fingers to toggle draw");
            }
        } else {
            snapDetector.stop();
            document.getElementById('mic-icon').className = 'fa-solid fa-microphone-slash text-white';
            showDynamicIslandMessage("Snap Detection Muted");
        }
    });

    // Auto-arm microphone on first user action
    document.addEventListener('click', () => {
        if (!snapDetector.isListening) {
            snapDetector.start();
        }
    }, { once: true });

    // Air draw button manual toggle
    airBtn.addEventListener('click', () => {
        airDrawEngine.toggle();
    });
}

function showDynamicIslandMessage(msg) {
    const island = document.getElementById('dynamic-island');
    const textEl = document.getElementById('island-main-text');
    const orig = textEl.textContent;
    textEl.textContent = msg;
    island.classList.add('expanded');
    setTimeout(() => {
        textEl.textContent = orig;
        if (!airDrawEngine.active) {
            island.classList.remove('expanded');
        }
    }, 2800);
}

// Chart.js initialization (NieR: Automata Tactical Telemetry Matrix)
function initChart() {
    const ctx = document.getElementById('engagementChart').getContext('2d');
    
    // NieR Ochre gradient fill (#C7AB69)
    const gradientOchre = ctx.createLinearGradient(0, 0, 0, 140);
    gradientOchre.addColorStop(0, 'rgba(199, 171, 105, 0.35)');
    gradientOchre.addColorStop(1, 'rgba(199, 171, 105, 0.0)');

    engagementChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(30).fill(''),
            datasets: [
                {
                    label: 'Engagement (%)',
                    data: Array(30).fill(100),
                    borderColor: '#C7AB69', // Pale Ochre
                    backgroundColor: gradientOchre,
                    fill: true,
                    tension: 0.15, // crisp tactical lines
                    borderWidth: 2,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'Attendance',
                    data: Array(30).fill(0),
                    borderColor: '#7D9B85', // Military Sage
                    borderWidth: 2,
                    tension: 0.1,
                    pointRadius: 0,
                    borderDash: [4, 4],
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: {
                    grid: { color: 'rgba(186, 178, 150, 0.08)' },
                    ticks: { display: false }
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(186, 178, 150, 0.1)' },
                    ticks: {
                        color: '#9C967E',
                        font: { family: "'Share Tech Mono', monospace", size: 10 },
                        callback: val => val + '%'
                    }
                },
                y1: {
                    position: 'right',
                    min: 0,
                    suggestedMax: 8,
                    grid: { display: false },
                    ticks: {
                        color: '#7D9B85',
                        font: { family: "'Share Tech Mono', monospace", size: 10 },
                        stepSize: 1
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(22, 21, 19, 0.95)',
                    titleColor: '#C7AB69',
                    bodyColor: '#DAD4BB',
                    borderColor: '#C7AB69',
                    borderWidth: 1,
                    cornerRadius: 0
                }
            }
        }
    });
}

// User Action Controls
function initControls() {
    const btnSim = document.getElementById('btn-src-sim');
    const btnCam0 = document.getElementById('btn-src-cam0');
    const btnUpload = document.getElementById('btn-src-upload');
    const fileInput = document.getElementById('video-file-input');

    const sourceButtons = [btnSim, btnCam0, btnUpload].filter(Boolean);
    function setActiveSource(activeBtn) {
        sourceButtons.forEach(b => b.classList.remove('active'));
        activeBtn.classList.add('active');
    }

    if (btnCam0) {
        btnCam0.addEventListener('click', () => {
            setActiveSource(btnCam0);
            switchSource('webcam0');
        });
    }

    if (btnSim) {
        btnSim.addEventListener('click', () => {
            setActiveSource(btnSim);
            switchSource('sim');
        });
    }

    if (btnUpload) {
        btnUpload.addEventListener('click', () => {
            fileInput.click();
        });
    }

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            setActiveSource(btnUpload);
            uploadVideoFile(fileInput.files[0]);
        }
    });

    // Heatmap Toggle (if present)
    const btnHeat = document.getElementById('btn-toggle-heat');
    if (btnHeat) {
        btnHeat.addEventListener('click', () => {
            isHeatmapActive = !isHeatmapActive;
            btnHeat.classList.toggle('nier-btn-primary', isHeatmapActive);
            btnHeat.classList.toggle('nier-btn-secondary', !isHeatmapActive);
            fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ show_heatmap: isHeatmapActive })
            });
        });
    }

    // Fullscreen Toggle
    const btnFullscreen = document.getElementById('btn-fullscreen');
    const viewportFrame = document.getElementById('viewport-frame');
    btnFullscreen.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            viewportFrame.requestFullscreen().catch(err => console.log(err));
        } else {
            document.exitFullscreen();
        }
    });

    // Report Export Button
    document.getElementById('btn-report-export').addEventListener('click', () => {
        exportSessionReport();
    });

    // Modal Close
    document.getElementById('modal-close-btn').addEventListener('click', closeModal);
    document.getElementById('modal-backdrop').addEventListener('click', closeModal);
}

function switchSource(sourceName) {
    fetch('/api/switch_source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: sourceName })
    }).then(res => res.json())
      .then(data => {
          showDynamicIslandMessage(`Switched to ${sourceName.toUpperCase()}`);
          const streamImg = document.getElementById('live-stream-img');
          streamImg.src = `/video_feed?t=${Date.now()}`;
      });
}

function uploadVideoFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    showDynamicIslandMessage(`Uploading ${file.name}...`);

    fetch('/api/upload_video', {
        method: 'POST',
        body: formData
    }).then(res => res.json())
      .then(data => {
          showDynamicIslandMessage(`Playing: ${file.name}`);
          const streamImg = document.getElementById('live-stream-img');
          streamImg.src = `/video_feed?t=${Date.now()}`;
      }).catch(err => {
          alert('Upload failed: ' + err);
      });
}

function exportSessionReport() {
    fetch('/api/export_report?format=json')
        .then(res => res.json())
        .then(data => {
            if (data.url) {
                const link = document.createElement('a');
                link.href = data.url;
                link.download = data.filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                showDynamicIslandMessage("Audit Report Downloaded");
            }
        });
}

// Telemetry Polling Loop
function startPollingTelemetry() {
    setInterval(() => {
        fetch('/api/telemetry')
            .then(res => res.json())
            .then(t => {
                // Update Top/Footer Bar safely
                const fpsEl = document.getElementById('hud-fps');
                if (fpsEl) fpsEl.textContent = `${t.fps || 0} FPS`;

                const devEl = document.getElementById('hud-device');
                if (devEl && t.device) {
                    devEl.textContent = t.device === '0' ? 'RTX 3050 CUDA' : 'CPU Optimized';
                }

                const backendEl = document.getElementById('hud-backend');
                if (backendEl && t.backend) {
                    backendEl.textContent = 'YOLO11-Pose Dual';
                }

                // Update KPI Cards: Attendance
                const headcount = t.headcount || 0;
                const headEl = document.getElementById('val-headcount');
                if (headEl) headEl.textContent = headcount;

                const peakEl = document.getElementById('val-peak');
                if (peakEl) {
                    const peak = t.peak_attendance !== undefined ? t.peak_attendance : headcount;
                    peakEl.textContent = `Peak: ${peak} seated`;
                }

                // Update KPI Cards: Attention Gauge
                const bar = document.getElementById('bar-attention');
                const valAtt = document.getElementById('val-attention');
                const statusEl = document.getElementById('val-attention-status');

                let att = 0;
                if (headcount > 0) {
                    const rawAtt = t.attention_index !== undefined ? t.attention_index : 95;
                    att = Math.max(0, Math.min(100, Math.round(Number(rawAtt))));
                    if (valAtt) valAtt.textContent = att;
                    if (bar) bar.style.width = `${Math.max(4, att)}%`;

                    // Dynamic colors based on engagement score
                    if (att >= 85) {
                        if (bar) bar.style.background = 'var(--nier-sage)';
                        if (valAtt) valAtt.className = 'kpi-val text-sage';
                    } else if (att >= 65) {
                        if (bar) bar.style.background = 'var(--nier-blue)';
                        if (valAtt) valAtt.className = 'kpi-val text-blue';
                    } else if (att >= 45) {
                        if (bar) bar.style.background = 'var(--nier-ochre)';
                        if (valAtt) valAtt.className = 'kpi-val text-amber';
                    } else {
                        if (bar) bar.style.background = 'var(--nier-alert-crimson)';
                        if (valAtt) valAtt.className = 'kpi-val text-crimson';
                    }

                    if (statusEl) {
                        statusEl.textContent = t.primary_status || 'Attentive & Focused';
                    }
                } else {
                    if (valAtt) {
                        valAtt.textContent = '0';
                        valAtt.className = 'kpi-val text-muted';
                    }
                    if (bar) {
                        bar.style.width = '0%';
                        bar.style.background = 'var(--nier-border)';
                    }
                    if (statusEl) {
                        statusEl.textContent = 'No Students Detected';
                    }
                }

                // Talking KPI Card
                const talkingCount = t.talking_count || 0;
                const valTalking = document.getElementById('val-talking');
                const talkingSub = document.getElementById('talking-sub-alert');
                if (valTalking) valTalking.textContent = talkingCount;
                if (talkingSub) {
                    if (talkingCount > 0) {
                        talkingSub.textContent = `Talking Active (${talkingCount} talking)`;
                        talkingSub.className = 'kpi-caption text-purple';
                    } else {
                        talkingSub.textContent = 'No active talking';
                        talkingSub.className = 'kpi-caption';
                    }
                }

                // Phone Alert
                const phoneCount = t.phone_count || 0;
                const valPhones = document.getElementById('val-phones');
                const alertBanner = document.getElementById('apple-alert-banner');
                const alertBannerText = document.getElementById('apple-alert-text');
                const phoneSub = document.getElementById('phone-sub-alert');

                if (valPhones) valPhones.textContent = phoneCount;
                if (phoneCount > 0) {
                    if (alertBanner) alertBanner.style.display = 'flex';
                    if (alertBannerText) alertBannerText.textContent = `Mobile Phone Detected (${phoneCount} active)`;
                    if (phoneSub) {
                        phoneSub.textContent = `Active Alert: ${phoneCount} in use`;
                        phoneSub.className = 'kpi-caption text-red';
                    }
                } else {
                    if (alertBanner) alertBanner.style.display = 'none';
                    if (phoneSub) {
                        phoneSub.textContent = 'No phones detected';
                        phoneSub.className = 'kpi-caption';
                    }
                }

                // Books & Notebooks count
                const valNotes = document.getElementById('val-notes');
                const bookSub = document.getElementById('book-sub-alert');
                const bookCount = t.book_count !== undefined ? t.book_count : (t.note_count || 0);
                if (valNotes) valNotes.textContent = bookCount;
                if (bookSub) {
                    bookSub.textContent = bookCount > 0 ? `${bookCount} item(s) in use` : 'No study materials';
                }

                // Head Down / Sleeping
                const sleepCount = t.sleeping_count || 0;
                const valSleep = document.getElementById('val-sleeping');
                const sleepSub = document.getElementById('sleep-sub-alert');
                if (valSleep) valSleep.textContent = sleepCount;
                if (sleepSub) {
                    if (sleepCount > 0) {
                        sleepSub.textContent = `Head Down Alert (${sleepCount} sleeping)`;
                        sleepSub.className = 'kpi-caption text-orange';
                    } else {
                        sleepSub.textContent = 'Normal posture';
                        sleepSub.className = 'kpi-caption';
                    }
                }

                // Update Chart
                if (engagementChart) {
                    const dataAtt = engagementChart.data.datasets[0].data;
                    const dataCount = engagementChart.data.datasets[1].data;
                    const labels = engagementChart.data.labels;

                    const plotAtt = headcount > 0 ? att : 0;
                    dataAtt.push(plotAtt);
                    dataCount.push(headcount);
                    labels.push(t.clock || '');

                    if (dataAtt.length > 30) {
                        dataAtt.shift();
                        dataCount.shift();
                        labels.shift();
                    }
                    engagementChart.update('none');
                }
            })
            .catch(err => console.error("Telemetry error:", err));
    }, 280);
}

// Incidents Polling Loop
function startPollingIncidents() {
    setInterval(() => {
        fetch('/api/incidents')
            .then(res => res.json())
            .then(incidents => {
                const container = document.getElementById('incidents-container');
                document.getElementById('incidents-total-badge').textContent = `${incidents.length} Events`;

                if (!incidents || incidents.length === 0) return;

                // Sound alert on new incident
                if (incidents.length > 0 && lastIncidentId !== incidents[0].id) {
                    lastIncidentId = incidents[0].id;
                    appleAudio.playAlertChime();
                    showDynamicIslandMessage(`Alert: ${incidents[0].type}`);
                }

                // Render Apple Incident Cards
                container.innerHTML = incidents.map(inc => `
                    <div class="incident-item" onclick="openModal('${inc.id}', '${inc.type}', '${inc.details}', '${inc.time}', '${inc.snapshot}')">
                        <div class="incident-top">
                            <span class="incident-type ${inc.severity === 'CRITICAL' ? 'text-red' : (inc.severity === 'WARNING' ? 'text-orange' : 'text-blue')}">
                                ● ${inc.type}
                            </span>
                            <span class="incident-time">${inc.time}</span>
                        </div>
                        <p class="incident-desc">${inc.details}</p>
                        <div class="incident-bottom">
                            <span>Student ${inc.student_id}</span>
                            <span class="evidence-tag"><i class="fa-solid fa-camera"></i> Inspect</span>
                        </div>
                    </div>
                `).join('');
            })
            .catch(err => console.error("Incident error:", err));
    }, 1000);
}

// Modal
function openModal(id, type, details, time, snapshotUrl) {
    document.getElementById('modal-title').textContent = `Evidence Record: ${id}`;
    document.getElementById('modal-evidence-img').src = snapshotUrl;
    document.getElementById('modal-details').innerHTML = `
        <div><strong>Type:</strong> ${type}</div>
        <div><strong>Time:</strong> ${time}</div>
        <div><strong>Log:</strong> ${details}</div>
    `;
    document.getElementById('evidence-modal').classList.add('active');
}

function closeModal() {
    document.getElementById('evidence-modal').classList.remove('active');
}
