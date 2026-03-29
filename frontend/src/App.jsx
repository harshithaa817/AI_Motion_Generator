import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Sparkles, Play, Pause, RefreshCw, Send, Move, Activity } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = "http://localhost:8000";

const KinematicChain = [
  [0, 2], [0, 1], [0, 3],                  // Lower body
  [1, 4], [4, 7], [7, 10],                 // Left leg
  [2, 5], [5, 8], [8, 11],                 // Right leg
  [3, 6], [6, 9], [9, 12],                 // Spine/Neck
  [9, 13], [13, 16], [16, 18], [18, 20],   // Left arm
  [9, 14], [14, 17], [17, 19], [19, 21],   // Right arm
];

function App() {
  const [prompt, setPrompt] = useState('a person walks in a circle');
  const [loading, setLoading] = useState(false);
  const [joints, setJoints] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  const generateMotion = async () => {
    if (!prompt) return;
    setLoading(true);
    setJoints(null);
    setCurrentFrame(0);
    try {
      const resp = await axios.post(`${API_BASE}/generate`, { prompt, num_frames: 120 });
      setJoints(resp.data.joints);
    } catch (err) {
      console.error(err);
      alert("Failed to generate motion. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (joints && playing) {
      const interval = setInterval(() => {
        setCurrentFrame(prev => (prev + 1) % joints[0][0].length);
      }, 50); // ~20fps
      return () => clearInterval(interval);
    }
  }, [joints, playing]);

  useEffect(() => {
    if (!joints || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    // Scale and center mapping
    const scale = 150;
    const offsetX = width / 2;
    const offsetY = height / 1.5;

    const frameData = joints.map(j => j.map(axis => axis[currentFrame]));
    
    // Draw connections
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    
    KinematicChain.forEach(([i, j]) => {
      const jointA = frameData[i];
      const jointB = frameData[j];
      
      // Simple orthographic projection (ignoring Z for now, using X and Y)
      // Note: HumanML3D often uses Z as up. Let's try X and Z for front view.
      ctx.beginPath();
      ctx.moveTo(jointA[0] * scale + offsetX, -jointA[1] * scale + offsetY);
      ctx.lineTo(jointB[0] * scale + offsetX, -jointB[1] * scale + offsetY);
      ctx.stroke();
    });

    // Draw joints
    ctx.fillStyle = '#ffffff';
    frameData.forEach(j => {
      ctx.beginPath();
      ctx.arc(j[0] * scale + offsetX, -j[1] * scale + offsetY, 4, 0, Math.PI * 2);
      ctx.fill();
    });

  }, [joints, currentFrame]);

  return (
    <div className="app-container">
      <header className="header fade-in">
        <div className="logo">MOTION.AI</div>
        <div className="status-badge glass-card">
          <Activity size={14} className="accent-text" />
          <span>GPU: CPU MODE</span>
        </div>
      </header>

      <main className="main-content">
        <section className="viewer-area glass-card fade-in" style={{ animationDelay: '0.1s' }}>
          <canvas 
            ref={canvasRef} 
            width={800} 
            height={600} 
            className="skeleton-viewer"
          />
          <AnimatePresence>
            {loading && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="overlay"
              >
                <div className="loading-spinner"></div>
                <p>Synthesizing Motion...</p>
              </motion.div>
            )}
          </AnimatePresence>
          {!joints && !loading && (
            <div className="placeholder-text">
              <Sparkles size={48} opacity={0.2} />
              <p>Enter a prompt to start</p>
            </div>
          )}
        </section>

        <section className="controls-area">
          <div className="glass-card fade-in" style={{ animationDelay: '0.2s' }}>
            <div className="input-group">
              <label className="input-label">Describe Motion</label>
              <textarea 
                className="text-input" 
                rows="3"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="e.g. a person performing a jumping jack"
              />
              <button 
                className="btn-primary"
                onClick={generateMotion}
                disabled={loading}
              >
                {loading ? <RefreshCw className="spin" size={18} /> : <Send size={18} />}
                {loading ? "Generating..." : "Generate Motion"}
              </button>
            </div>
          </div>

          <div className="glass-card fade-in" style={{ animationDelay: '0.3s' }}>
            <div className="input-label">Playback</div>
            <div className="playback-controls">
              <button className="icon-btn" onClick={() => setPlaying(!playing)}>
                {playing ? <Pause size={20} /> : <Play size={20} />}
              </button>
              <div className="frame-info">
                Frame {currentFrame}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
