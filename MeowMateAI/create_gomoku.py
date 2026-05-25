from pathlib import Path


# 使用相对路径，让项目更易移植
# 默认在脚本同级目录下创建游戏文件夹
PROJECT_DIR = Path(__file__).parent.parent / "GomokuGame"

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>五子棋</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <div class="app">
    <header class="panel topbar">
      <div>
        <h1>五子棋</h1>
        <p>本地双人对战，黑子先手，先连成五子者获胜。</p>
      </div>
      <div class="side-info">
        <div id="status" class="status">当前回合：黑子</div>
        <div class="actions">
          <button id="restartBtn">重新开始</button>
          <button id="undoBtn">悔棋一步</button>
        </div>
      </div>
    </header>

    <main class="board-panel">
      <canvas id="board" width="600" height="600" aria-label="五子棋棋盘"></canvas>
    </main>

    <section class="panel tips">
      <h2>玩法说明</h2>
      <ul>
        <li>点击棋盘交叉点即可落子。</li>
        <li>黑白双方轮流下子，横向、纵向或斜向先连成 5 子即可获胜。</li>
        <li>支持悔棋和重新开始。</li>
      </ul>
    </section>
  </div>
  <script src="script.js"></script>
</body>
</html>
"""

STYLES_CSS = """:root {
  --bg1: #f6ead1;
  --bg2: #d6b184;
  --panel: rgba(255, 248, 235, 0.92);
  --text: #2f2419;
  --accent: #8b5a2b;
  --shadow: 0 12px 30px rgba(0, 0, 0, 0.14);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "Microsoft YaHei", sans-serif;
  color: var(--text);
  background: radial-gradient(circle at top, var(--bg1), var(--bg2));
}

.app {
  width: min(920px, calc(100vw - 32px));
  margin: 24px auto;
}

.panel {
  background: var(--panel);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 20px 24px;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}

h1, h2 {
  margin: 0 0 8px;
}

p {
  margin: 0;
}

.side-info {
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: flex-end;
}

.status {
  font-size: 1.08rem;
  font-weight: 700;
}

.actions {
  display: flex;
  gap: 12px;
}

button {
  border: none;
  border-radius: 999px;
  background: var(--accent);
  color: #fff;
  padding: 10px 18px;
  cursor: pointer;
  font-size: 0.95rem;
  transition: transform 0.15s ease, opacity 0.15s ease;
}

button:hover {
  transform: translateY(-1px);
  opacity: 0.92;
}

.board-panel {
  display: flex;
  justify-content: center;
  margin-bottom: 20px;
}

canvas {
  background: linear-gradient(135deg, #deb887, #c89b68);
  border-radius: 22px;
  box-shadow: var(--shadow);
  max-width: 100%;
  height: auto;
}

.tips ul {
  margin: 0;
  padding-left: 20px;
}

.tips li + li {
  margin-top: 8px;
}

@media (max-width: 700px) {
  .topbar {
    flex-direction: column;
    align-items: flex-start;
  }

  .side-info {
    align-items: flex-start;
  }

  .actions {
    flex-wrap: wrap;
  }
}
"""

SCRIPT_JS = """const boardSize = 15;
const cellSize = 40;
const padding = 20;

const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const restartBtn = document.getElementById("restartBtn");
const undoBtn = document.getElementById("undoBtn");

let board = [];
let currentPlayer = 1;
let gameOver = false;
let history = [];

function initBoard() {
  board = Array.from({ length: boardSize }, () => Array(boardSize).fill(0));
  currentPlayer = 1;
  gameOver = false;
  history = [];
  updateStatus();
  drawBoard();
}

function updateStatus(message) {
  statusEl.textContent = message || `当前回合：${currentPlayer === 1 ? "黑子" : "白子"}`;
}

function drawBoard() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#6d4c2f";
  ctx.lineWidth = 1;

  for (let i = 0; i < boardSize; i += 1) {
    const offset = padding + i * cellSize;
    ctx.beginPath();
    ctx.moveTo(padding, offset);
    ctx.lineTo(canvas.width - padding, offset);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(offset, padding);
    ctx.lineTo(offset, canvas.height - padding);
    ctx.stroke();
  }

  [3, 7, 11].forEach((row) => {
    [3, 7, 11].forEach((col) => {
      ctx.beginPath();
      ctx.arc(padding + col * cellSize, padding + row * cellSize, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#6d4c2f";
      ctx.fill();
    });
  });

  board.forEach((row, y) => {
    row.forEach((cell, x) => {
      if (cell) drawStone(x, y, cell);
    });
  });
}

function drawStone(x, y, player) {
  const px = padding + x * cellSize;
  const py = padding + y * cellSize;
  const gradient = ctx.createRadialGradient(px - 6, py - 6, 4, px, py, 18);

  if (player === 1) {
    gradient.addColorStop(0, "#666");
    gradient.addColorStop(1, "#000");
  } else {
    gradient.addColorStop(0, "#fff");
    gradient.addColorStop(1, "#d9d9d9");
  }

  ctx.beginPath();
  ctx.arc(px, py, 16, 0, Math.PI * 2);
  ctx.fillStyle = gradient;
  ctx.fill();
  ctx.strokeStyle = player === 1 ? "#111" : "#999";
  ctx.stroke();
}

function getPosition(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clickX = (event.clientX - rect.left) * scaleX;
  const clickY = (event.clientY - rect.top) * scaleY;

  const x = Math.round((clickX - padding) / cellSize);
  const y = Math.round((clickY - padding) / cellSize);

  if (x < 0 || x >= boardSize || y < 0 || y >= boardSize) return null;

  const targetX = padding + x * cellSize;
  const targetY = padding + y * cellSize;
  if (Math.hypot(clickX - targetX, clickY - targetY) > cellSize * 0.45) return null;

  return { x, y };
}

function countInDirection(x, y, dx, dy, player) {
  let total = 0;
  let cx = x + dx;
  let cy = y + dy;

  while (cx >= 0 && cx < boardSize && cy >= 0 && cy < boardSize && board[cy][cx] === player) {
    total += 1;
    cx += dx;
    cy += dy;
  }

  return total;
}

function checkWin(x, y, player) {
  const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
  return directions.some(([dx, dy]) => 1 + countInDirection(x, y, dx, dy, player) + countInDirection(x, y, -dx, -dy, player) >= 5);
}

function handleBoardClick(event) {
  if (gameOver) return;

  const pos = getPosition(event);
  if (!pos) return;

  const { x, y } = pos;
  if (board[y][x] !== 0) return;

  board[y][x] = currentPlayer;
  history.push({ x, y, player: currentPlayer });
  drawBoard();

  if (checkWin(x, y, currentPlayer)) {
    gameOver = true;
    updateStatus(`${currentPlayer === 1 ? "黑子" : "白子"} 获胜！`);
    return;
  }

  if (history.length === boardSize * boardSize) {
    gameOver = true;
    updateStatus("平局，棋盘已满。");
    return;
  }

  currentPlayer = currentPlayer === 1 ? 2 : 1;
  updateStatus();
}

function undoMove() {
  if (history.length === 0) return;

  const last = history.pop();
  board[last.y][last.x] = 0;
  currentPlayer = last.player;
  gameOver = false;
  drawBoard();
  updateStatus();
}

canvas.addEventListener("click", handleBoardClick);
restartBtn.addEventListener("click", initBoard);
undoBtn.addEventListener("click", undoMove);

initBoard();
"""


def main() -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (PROJECT_DIR / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (PROJECT_DIR / "script.js").write_text(SCRIPT_JS, encoding="utf-8")
    for item in sorted(PROJECT_DIR.iterdir()):
        print(item.name)


if __name__ == "__main__":
    main()