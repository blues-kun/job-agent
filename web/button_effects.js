// 🎨 匹配职位按钮特效系统

// 涟漪效果
function addRippleEffect(button) {
  button.classList.add('ripple');
  setTimeout(() => {
    button.classList.remove('ripple');
  }, 600);
}

// 粒子爆炸效果
function createParticles(x, y) {
  const container = document.createElement('div');
  container.className = 'magic-particles';
  document.body.appendChild(container);
  
  const particleCount = 12;
  const colors = ['#a78bfa', '#8b5cf6', '#c4b5fd', '#ddd6fe'];
  
  for (let i = 0; i < particleCount; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    const angle = (Math.PI * 2 * i) / particleCount;
    const velocity = 80 + Math.random() * 40;
    const tx = Math.cos(angle) * velocity;
    const ty = Math.sin(angle) * velocity;
    
    particle.style.left = x + 'px';
    particle.style.top = y + 'px';
    particle.style.background = colors[i % colors.length];
    particle.style.setProperty('--tx', tx + 'px');
    particle.style.setProperty('--ty', ty + 'px');
    
    container.appendChild(particle);
  }
  
  setTimeout(() => {
    container.remove();
  }, 1000);
}

// 脉冲发光效果
function addPulseGlow(button) {
  button.classList.add('active');
  setTimeout(() => {
    button.classList.remove('active');
  }, 600);
}

// 震动效果（如果支持）
function vibrate() {
  if (navigator.vibrate) {
    navigator.vibrate([50, 30, 50]);
  }
}

// 主特效函数
function triggerMatchButtonEffects(event) {
  const button = event.currentTarget;
  const rect = button.getBoundingClientRect();
  const x = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;
  
  // 1. 涟漪效果
  addRippleEffect(button);
  
  // 2. 粒子爆炸
  createParticles(x, y);
  
  // 3. 脉冲发光
  addPulseGlow(button);
  
  // 4. 震动反馈（移动设备）
  vibrate();
  
  // 5. 添加加载状态
  button.classList.add('loading');
  button.disabled = true;
  
  // 返回清理函数
  return () => {
    button.classList.remove('loading');
    button.disabled = false;
  };
}

// 导出供app.js使用
window.triggerMatchButtonEffects = triggerMatchButtonEffects;

