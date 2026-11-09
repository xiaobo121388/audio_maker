# mcsfx

面向 Agent 的游戏录像音效生成 CLI。输入一段本地 Minecraft 短片，显式选择 Grok 首帧引导生成或 Grok / Seedance 视频编辑接口，保存模型有声视频，也可以把生成的声音合回原录屏。

**本工具不负责进入游戏、录制、剪切独立音效或修改资源包。** 这些环节由 Agent 根据具体模组执行。视频时长接近不等于声音事件对齐，生成结果仍需实际试听。

多阶段技能默认采用“完整母带优先”，不要为每个阶段独立生成、裁剪再拼接。跨段淡化不能保证音色统一。详见 [Agent 整段音效工作流](docs/agent-soundtrack-workflow.md)，包含模型时长限制、整体声音设计、一次播放、晚加入后缀与试听门禁。

## 安装

需要 Python 3.12+、uv，以及包含 `ffmpeg` 和 `ffprobe` 的 FFmpeg 安装。工具不会自行安装 FFmpeg，也不会创建云存储桶。

```powershell
uv sync --locked --group dev
uv run --locked mcsfx --help
uv run --locked mcsfx doctor --profile grok --json
```

也可用 `uv tool install . --python 3.12` 安装独立命令。仓库通过 `.python-version` 使用 Python 3.12，通过 `uv.lock` 固定依赖。

FFmpeg 不在 PATH 时，用环境变量指定两个可执行文件；不能把额外命令行参数放入路径：

```powershell
$env:MCSFX_FFMPEG = 'C:\tools\ffmpeg\bin\ffmpeg.exe'
$env:MCSFX_FFPROBE = 'C:\tools\ffmpeg\bin\ffprobe.exe'
```

## 配置

不提供配置文件时内置 `grok`、`grok-sfx` 和 `seedance` 三个 profile。自定义配置默认读取当前目录 `mcsfx.toml`，也可用 `--config` 或 `MCSFX_CONFIG` 显式指定。示例文件是 `mcsfx.example.toml`，示例桶名必须替换成自己的已有私有桶。

凭据只从环境变量读取，不把密钥直接写进 TOML、提示词或命令参数：

```powershell
$env:XAI_API_KEY = '<从你的密钥管理器读取>'
$env:ARK_API_KEY = '<从你的密钥管理器读取>'
$env:TOS_ACCESS_KEY_ID = '<从你的密钥管理器读取>'
$env:TOS_SECRET_ACCESS_KEY = '<从你的密钥管理器读取>'
# 使用临时身份时另设 TOS_SECURITY_TOKEN。
```

`--config`、`--state-dir`、`--json` 放在子命令之后。环境变量只影响当前 PowerShell 会话；不要将真实密钥提交到版本控制。

### 官方和中转

| Profile | 协议 / 能力预设 | 初始模型 | 实际提交素材 |
| --- | --- | --- | --- |
| grok | xai / grok-edit | grok-imagine-video | data_uri；可改为 tos |
| grok-sfx | xai / grok-keyframe | grok-imagine-video | 首帧 PNG data_uri；不支持 tos |
| seedance | ark / seedance-2.5-edit | doubao-seedance-2-5-260628 | tos |

中转必须完整兼容所选官方协议：路径、鉴权、请求字段、任务查询及返回结构。`base_url` 包含 API 前缀，例如 xAI 的 `/v1` 或 Ark 的 `/api/v3`。不能仅因为服务商写着“OpenAI 兼容”就认为能用。

可以增加具名 profile，指定中转地址、模型别名和独立的密钥环境变量名称。别名必须映射到同一能力预设；不能将 Seedance 2.0 别名填进 2.5 编辑预设，也不自动发现或猜测其他协议。只接受 HTTPS 地址，不允许在地址内嵌入账号、密码或 API key 查询参数。

模型接口及视频下载使用直连 HTTP 客户端，不读取 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 或环境 CA 覆盖。第一版没有 CLI 代理选项；需要网络转发时配置可直连的兼容中转。TOS 使用其官方 SDK 的网络行为。

### 首帧音效生成与认证下载

`grok-sfx` 是新增的显式选择，CLI 默认 profile 仍为 `grok`，旧 `grok-edit` 不会被自动替换。本项目此前的 Grok 编辑实测返回了无音轨视频，因此本预设改走首帧图生视频请求；这不保证所有生成结果都有有效音效，下载后仍需完整解码和非全静音验证。

调用者仍提供本地 MP4/MOV。工具使用配置的 FFmpeg 无损复制视频轨生成静音 `input.mp4`，校验视频包和时间戳不变，再提取首个解码帧为 PNG 并验证尺寸、完整解码和大小。实际请求为：

```text
POST <base_url>/videos/generations
```

```json
{
  "model": "grok-imagine-video",
  "prompt": "首帧引导生成的默认音效要求与用户动作说明",
  "image": {"url": "data:image/png;base64,..."},
  "duration": 6,
  "resolution": "480p",
  "generate_audio": true
}
```

`duration` 为视频轨真实时长向上取整的整数，范围 1–15；例如 5.1 秒请求 6 秒，2 秒请求 2 秒。超过 15 秒直接拒绝，不自动裁剪、变速或压缩时间轴。`grok-keyframe` 是本地预设名，不会发送新版 API 的 `keyframes` 字段，也不会发送整段视频。

**推荐 `--mode regenerate`。** 只有首帧外观被提供给模型，原录像中的动作和时序并未提交；所有此类任务持续报告 `keyframe_guided_motion_unverified` 和 `alignment_unverified`。截图编码成静态 MP4 也可作为参考，但应在任务说明和提示词中如实标记为静态参考，不称为动作录像或已对齐的音效。`preserve` 不放宽原有 0.5 秒限制，也不自动对齐事件；使用该模式还会提示 `regenerate_recommended`。

需下载认证内容端点的中转，必须显式启用以下配置。内置 profile 可以只覆盖部分字段，完整写法为：

```toml
[profiles.grok-sfx]
protocol = "xai"
preset = "grok-keyframe"
base_url = "https://ai.xiaobo12138.site/v1"
model = "grok-imagine-video"
api_key_env = "XAI_API_KEY"
transport = "data_uri"
download_mode = "same_origin_content"
```

- `download_mode` 默认 `public`，旧任务快照缺少此字段时也按 `public` 处理，公共 HTTPS 下载行为不变。
- `same_origin_content` 仅接受 xAI 协议。查询结果的 `video.url` 只能是配置的 `base_url` 去掉末尾 `/` 后拼接 `/videos/<已保存远端ID>/content`，或该 URL 的完整绝对路径部分。API 前缀必须精确匹配，不接受任意相对地址、其他任务 ID、查询参数、片段、编码路径别名、路径穿越或其他主机/端口。
- 只有这个精确初始地址携带 Bearer。每次重定向均使用原有安全下载检查且不携带 Authorization、客户端默认凭据或 Cookie，即使重定向回原地址也一样。该模式不能直接将任意 CDN URL 当作认证首跳，服务商可以重定向至公共 HTTPS CDN。
- 仍保留最多 3 次下载重试、5 次重定向、1 GiB 大小上限及原子不覆盖发布。密钥仅在运行时从环境变量读取，不写入任务文件、产物清单或错误输出。

### TOS

Seedance 先将静音副本上传到配置中的私有桶，签发临时 GET 链接，提交给模型。`grok-edit` 默认提交视频 Base64，不使用 TOS；不接受视频 data URI 的中转可显式改为 `transport = "tos"`。`grok-keyframe` 仅支持 PNG data URI，不经过 TOS。

配置包含 `endpoint`、`region`、`bucket`、`prefix` 以及凭据环境变量名称。对象位于 `prefix/<job-id>/input.mp4`；只操作本任务记录的对象，不创建桶、改变 ACL 或遍历删除桶内容。

预签名链接默认 96 小时。**临时身份凭据自身过期可能使链接提前失效**，必须让凭据有效期覆盖排队和生成时间。网络策略也必须允许模型抓取素材。

任务确认远端终态后自动清理输入对象；本地等待超时、进程中断、提交结果不明确时保留。自动删除失败不会丢弃成功产物，会返回 `cleanup_pending`；之后执行 `cleanup`。存储费、访问费和必要的桶生命周期规则由账号所有者管理，本工具不自动修改生命周期配置。

## Agent 工作流

1. Agent 核对游戏和项目身份，按现有 MCDK 测试规范录制一次完整动作，留出短收尾；录制和游戏状态清理由 Agent 负责。
2. 准备 MP4/MOV、H.264/H.265 短片。若要比较两个默认模型，使用同一段 4–8 秒素材。
3. 运行 `doctor` 和 `generate --dry-run` 检查依赖、输入和配置；根据动作描述补充提示词。
4. 提交一次，保存返回的 job ID；使用 `fetch` 等待、下载，不反复执行 `generate` 来查询进度。
5. 对照原录屏与模型视频试听，检查蓄力、释放、命中和收尾位置，有无人声、音乐、漏声或不相关声响。
6. Agent 自行提取、裁剪和对齐音效，按目标工程的音频格式及播放接口接入，再进行真实游戏测试。

## 命令

```powershell
# 无网络检查：默认报告全部 profile；指定 profile 可聚焦单一后端。
uv run --locked mcsfx doctor --profile grok --json

# 不上传、不生成、不创建任务目录。TOS 仍需填写有效的非秘密配置。
uv run --locked mcsfx generate '.\技能录像.mp4' --profile grok --dry-run --json

# 首帧方案的 dry-run 会真实准备静音视频并提取/验证 PNG，然后删除临时文件。
uv run --locked mcsfx generate '.\技能录像.mp4' --profile grok-sfx --mode regenerate --dry-run --json

# 首帧引导音效生成；动作经过由提示词描述，不能保证复现原录像时序。
uv run --locked mcsfx generate '.\技能录像.mp4' --profile grok-sfx --mode regenerate --prompt '静态游戏参考画面；生成两次金属撞击，无人声无音乐' --out '.\outputs' --wait --json

# 等待有声结果，默认 preserve 模式将声音合回原录屏。
uv run --locked mcsfx generate '.\技能录像.mp4' --profile grok --prompt '蓄力低鸣，随后喷射火焰，最后爆炸命中' --out '.\outputs' --wait --json

# 异步提交；regenerate 直接使用模型生成的画面和声音。
uv run --locked mcsfx generate '.\技能录像.mp4' --profile seedance --config '.\mcsfx.toml' --mode regenerate --out '.\outputs' --json

# 后续命令使用相同的 state-dir；凭据从提交时记录的环境变量名称读取。
uv run --locked mcsfx status <job-id> --json
uv run --locked mcsfx fetch <job-id> --wait --timeout 900 --json
uv run --locked mcsfx cleanup <job-id> --json
```

`generate` 每次执行都是一个新付费任务，不会自动缓存或去重。`--wait` 默认最多等待 600 秒，可以调整；本地超时并不取消远端生成。查询、下载有最多三次的退避重试；创建请求绝不自动重试。

### 输入和声音规则

- Grok 编辑预设最大输入 8.7 秒，不向该接口发送其不接受的时长、比例或分辨率参数；data URI 的本地输入大小限制为 45 MiB，预留 Base64 和 JSON 开销。
- Grok 首帧预设接受大于 0 且不超过 15 秒的本地视频；本地输入和提取的 PNG 各自限制 45 MiB。请求时长向上取整，不自动改变输入视频；不套用 Seedance 的尺寸、帧率限制。
- Seedance 2.5 编辑预设接受 4–30 秒、24–60 FPS，最大 200 MB；宽高各 300–6000 像素、宽高比 0.4–2.5、总像素 407696–8295044。超限会报错，不自动剪切或缩放。
- 上传副本会移除全部已有音轨；原文件不会被修改。若希望模型使用原录音作为参考，当前版本不支持。
- 编辑预设默认提示词要求保持画面和动作时序；首帧预设明确只提供首帧，要求按描述生成动作及音效。两者都不添加音乐、人声或旁白；`--prompt` 添加具体说明，不取消这些默认要求。
- `preserve` 使用原视频轨，不用模型生成画面。音轨时长差不超过 0.5 秒时，仅对声音尾部补静音或裁尾；不会变速，也不会自动对齐动作事件。
- 超过 0.5 秒报告 `TIMELINE_MISMATCH`，模型原片仍保留。可以由 Agent 从该文件提取和手动对齐声音，不需要重新付费生成。
- 下载及合回后执行完整解码、音轨存在性和全静音检查。这些不检测语音、音乐或语义质量，所有成功结果仍包含 `alignment_unverified`。

## 输出与恢复

每个任务使用独立 UUID 子目录，默认位于 `outputs/<job-id>/`：

| 文件 | 含义 |
| --- | --- |
| manifest.json | 非秘密的任务快照、提示词、远端 ID、状态、摘要、警告及产物清单 |
| input.original.mp4 / .mov | 独立输入快照，保证原录像移动后仍可合回 |
| input.mp4 | 无音轨副本；编辑预设提交整段，首帧预设只在本地用于提取 |
| keyframe.png | 仅首帧预设：经过校验的首个解码帧，作为 PNG data URI 提交 |
| generated.mp4 | 下载的模型有声视频；不会被合回操作覆盖 |
| result.mp4 | preserve 的合回结果；regenerate 的结果路径直接指向 generated.mp4 |

任务索引默认位于 Windows `%LOCALAPPDATA%\mcsfx\jobs`，其他系统为 `~/.local/state/mcsfx/jobs`。`--state-dir` 优先于 `MCSFX_STATE_DIR` 和 TOML 设置。使用不同状态目录无法查询原任务。

`--json` 只在 stdout 输出一个 JSON 对象；运行进度发往 stderr。`remote_id` 同时返回服务商任务标识，便于异常时人工核对账号。核心字段：

```json
{
  "schema_version": 1,
  "ok": true,
  "job_id": "任务 UUID",
  "remote_id": "服务商任务标识",
  "status": "completed",
  "remote_status": "succeeded",
  "artifacts": {"manifest": "绝对路径", "generated": "绝对路径", "result": "绝对路径"},
  "warnings": ["alignment_unverified"],
  "metrics": {},
  "error": null
}
```

`ok=true` 表示本次命令成功，不一定代表已生成完成。只有 `status=completed` 表示本地技术验收完成；`queued/running` 仍在执行，`ready` 表示远端成功但尚未完成本地处理，`local_failed` 表示下载或媒体处理失败。

`metrics` 包含模型视频时长、模型视频与输入视频时长差、合回后的时长差，以及声音尾部调整量。`audio_adjustment_seconds > 0` 为补静音，`< 0` 为裁尾。最终文件和输入快照保存 SHA-256 校验和。

首帧任务还记录 `generation` 中的整数时长、分辨率、音轨开关，以及 `keyframe` 元数据、SHA-256 和 `artifacts.keyframe` 路径；不保存 Base64 或密钥。恢复时沿用提交时记录的 profile、下载模式和远端 ID，不重新提取或提交。`status`、`fetch` 都不会重新生成；多个独立任务的调度和并发上限由调用者控制。

| 情况 | 处理方式 |
| --- | --- |
| WAIT_TIMEOUT / Ctrl+C | 后续运行 `fetch --wait`；不假定远端已取消 |
| SUBMISSION_UNKNOWN | 先检查服务商账号；不能确认是否收费/生成，工具不会自动重发或删除输入 |
| 查询、下载暂时失败 | 再次 `fetch`，沿用原任务 ID |
| TIMELINE_MISMATCH | 使用已保留的 generated 文件，由 Agent 重新对齐 |
| cleanup_pending | 使用 `cleanup` 重试；只删本任务 TOS 对象 |
| JOB_BUSY | 同一任务有其他进程操作；等待其结束，不能手动删锁文件绕过 |
| ARTIFACT_CHANGED / ARTIFACT_CONFLICT | 文件被修改或冲突，拒绝覆盖；检查任务目录 |
| AUTH_ERROR / CONFIG_ERROR | 检查环境变量和配置，不在日志中打印密钥 |

退出码：0 成功，1 操作失败，2 参数错误，3 本地等待超时，130 用户中断。JSON 错误包含稳定 `code`、可显示 `message`、`retryable` 和 `details`。`retryable` 表示同一任务的操作可重试，不代表应再次运行 `generate`。

状态快照原子写入，操作持有操作系统文件锁，进程退出会释放锁。建议将状态目录和产物放在本地 NTFS；产物原子发布使用硬链接，不支持硬链接的文件系统会安全失败，不退化为覆盖文件。

## 测试

```powershell
uv run --locked pytest -q -m "not live"
uv run --locked ruff check src tests
```

默认测试不访问付费模型、不上传 TOS。协议测试使用模拟 HTTP 传输；TOS 测试使用模拟 SDK 客户端。媒体测试使用本机 FFmpeg；缺失时明确 skip，不算媒体验收通过。

付费实测需要显式启用，提供一段真实 4–8 秒游戏录像，并配置每个后端的密钥。默认每个 profile 提交一次，测试同时保留模型原片和合回结果：

```powershell
$env:MCSFX_LIVE_TESTS = '1'
$env:MCSFX_LIVE_VIDEO = 'D:\recordings\attack.mp4'
$env:MCSFX_LIVE_CONFIG = 'D:\configs\mcsfx.toml'
$env:MCSFX_LIVE_PROFILES = 'grok,seedance'
uv run --locked pytest -m live -q
Remove-Item Env:MCSFX_LIVE_TESTS
```

中转需加入自己的 profile 名称单独验证。真实测试可能收费，也可能因模型权限、审核或服务商能力变化失败；没有运行真实测试时不能声明对应服务可用或音效达到质量要求。

实测产物默认保留在 `outputs/live`，任务索引保留在 `outputs/live/.state`。实测等待中断时，使用该目录作为 `fetch --state-dir`，不要重新运行整项付费测试。普通任务的 manifest 也记录 `state_dir`，方便定位恢复入口。

## 接口依据

当前能力预设仅覆盖已选模型，账号可用性和服务限制以服务商为准：

- 2026-11-09 核对 xAI Generation / Image-to-Video 官方 Markdown 文档中的 `image` data URI、1–15 秒、480p 和音轨开关。新预设仍按本项目约定使用 `grok-imagine-video`，不自动升级模型；中转认证内容端点属于显式兼容选项，不将其视为官方公共下载行为。开发验证为离线 FFmpeg 和 mock，不等于本次已完成付费服务实测。
- xAI Video Generation: https://docs.x.ai/developers/model-capabilities/video/generation.md
- xAI Image-to-Video: https://docs.x.ai/developers/model-capabilities/video/image-to-video.md
- xAI Video Editing: https://docs.x.ai/developers/model-capabilities/video/editing
- xAI Videos REST: https://docs.x.ai/developers/rest-api-reference/inference/videos
- Seedance 2.5: https://docs.volcengine.com/docs/ark/seedance-2-5?lang=zh
- Ark 创建视频任务: https://docs.volcengine.com/docs/ark/create-video-generation-task-api?lang=zh
- TOS Python 预签名: https://docs.volcengine.com/docs/torch-object-storage/generalpresigned-python-sdk?lang=en
