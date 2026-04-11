# ComfyUI Nodes 2.0 & Schema V3 Github Copilot Instructions

You are an expert developer specializing in the modern ComfyUI Nodes 2.0 architecture and the Schema V3 (`comfy_api.latest`) Python backend. When generating, reviewing, or refactoring code for this project, you must strictly adhere to the following declarative, stateless, and Vue 3-compatible guidelines. Do not generate legacy V1 immediate-mode Canvas code or stateful Python classes.

## 0. Development Environment Context

**IMPORTANT:** This repository is NOT part of an active ComfyUI installation. It is a standalone development workspace exclusively for creating and testing new custom nodes for the Custom Node Pack.

### Reference ComfyUI Installation
A complete ComfyUI installation is available at `D:\comfy_for_agent` for research and reference purposes. When developing custom nodes or researching the V3 Schema API:

* **Primary API Reference:** `D:\comfy_for_agent\ComfyUI\comfy_api\latest\_io.py` contains the authoritative Schema V3 type definitions, input/output specifications, and the complete `io` module interface for Node 2.0 development.
* Use this reference installation to verify API signatures, explore available input types, and understand the complete V3 architecture.
* All nodes developed in this workspace should conform to the patterns and types defined in the reference installation's `comfy_api.latest` module.

---

## 1. Project Structure & Naming Conventions

### File Organization
* Each node lives in its own file under `nodes/` (e.g., `nodes/my_new_node.py`).
* All node classes are registered in `nodes/__init__.py` via the `NODE_LIST` array.
* The root `__init__.py` contains the `DuffyNodesExtension(ComfyExtension)` class and `comfy_entrypoint()`.
* Frontend code is developed as Vue 3 Single-File Components (SFCs) in `src/`.
* We use Vite to build the frontend. The compiled JS output goes to the `web/` directory.
* The root `WEB_DIRECTORY = "./web"` serves the entire `web/` folder to the frontend.

### Naming Conventions
* **Python class names:** `Duffy<PascalCaseName>` (e.g., `DuffyImageStitch`, `DuffyFloatMath`).
* **Schema `node_id`:** `"Duffy_<PascalCaseName>"` (e.g., `"Duffy_ImageStitch"`, `"Duffy_FloatMath"`).
* **Schema `category`:** `"Duffy/<Subcategory>"` (e.g., `"Duffy/Image"`, `"Duffy/Math"`, `"Duffy/Latent"`).
* **JS extension names:** `"Duffy.<PascalCaseName>"` (e.g., `"Duffy.ImageStitch.Vue"`).
* **Python filenames:** `snake_case.py` matching the node's purpose.

---

## 2. Python Backend Rules (Schema V3)

### Core Architecture
* Inherit all custom nodes strictly from the `io.ComfyNode` base class.
* Ensure all nodes are completely stateless; do not utilize `__init__` methods to store instance variables or configurations.
* Define node inputs, outputs, and metadata declaratively using `@classmethod def define_schema(cls) -> io.Schema`.
* Contain all logic within a `@classmethod def execute(cls, ...)` method.
* Always accept `**kwargs` in `execute()` to future-proof against additional framework-injected parameters.

### Schema Definition — List-Style Inputs (MANDATORY)
All schemas in this project use **list-style** inputs and outputs with the `id` as the first positional argument. Do NOT use dict-style `inputs={}` — use `inputs=[]`.

### Image Tensor Format (CRITICAL)
* **ComfyUI image tensors are `[B, H, W, C]`** (batch, height, width, channels), NOT `[B, C, H, W]`.
* Values are float32 normalized between `0.0` and `1.0`.
* **`comfy.utils.common_upscale()` expects `[B, C, H, W]`** — convert with `.movedim(-1, 1)` before calling and `.movedim(1, -1)` after.

### Mask Tensor Format
* Masks are `[B, H, W]` (no channel dimension), float32 in `[0.0, 1.0]`.
* Alpha channel extraction inverts: `mask = 1.0 - alpha` (black = opaque in ComfyUI).

### Cache Control: `fingerprint_inputs`
Replaces legacy `IS_CHANGED`. Return a deterministic value — same return = cache hit, different = re-execute.

### Input Validation: `validate_inputs`
Return `True` for valid inputs or an error string to block execution.

---

## 3. Frontend & Styling Rules (Vue 3 SFC & Vite Architecture)

We build modern frontend widgets using **Vue 3 Single-File Components (.vue)** and compile them into native ComfyUI extensions using **Vite**. 

### Vite Build Setup
All frontend code lives in `src/`. We use Vite with `@vitejs/plugin-vue` and `vite-plugin-css-injected-by-js` to compile the Vue component and its scoped CSS into a single standalone `.js` file in the `web/` folder. ComfyUI's internal scripts (`app.js`, `api.js`) must be externalized.

### Vue App Mounting & Cleanup (CRITICAL)
To inject custom HTML/Vue apps into nodes, use `node.addDOMWidget()`. You **MUST** ensure the Vue instance is properly unmounted when the node is deleted to prevent severe memory leaks.

```javascript
import { createApp } from "vue";
import MyVueWidget from "./MyVueWidget.vue";

async nodeCreated(node) {
    if (node.comfyClass !== "Duffy_MyNode") return;

    const container = document.createElement("div");
    container.style.cssText = "width:100%; box-sizing:border-box; overflow:hidden;";

    // Prevent canvas drag/zoom when interacting with the widget
    container.addEventListener("pointerdown", (e) => e.stopPropagation());
    container.addEventListener("wheel", (e) => e.stopPropagation());

    // Mount the Vue App
    const vueApp = createApp(MyVueWidget, {
        onChange: (data) => {
            node.setDirtyCanvas(true, true);
        }
    });
    const instance = vueApp.mount(container);

    // Register the DOM widget
    const domWidget = node.addDOMWidget("my_vue_ui", "custom", container, { serialize: false });
    domWidget.computeSize = () => [300, 200];

    // CRITICAL: Prevent Memory Leaks! Unmount Vue when node is removed
    const origRemoved = node.onRemoved;
    node.onRemoved = function () {
        instance.cleanup?.(); // Custom cleanup inside the Vue component
        vueApp.unmount();     // Destroy the Vue instance
        origRemoved?.apply(this, arguments);
    };
}
```

### The Advanced Hidden Widget Pattern (JSON State Sync)
To save complex Vue state in ComfyUI workflows, we map the Vue app's state to a hidden io.String.Input(socketless=True) defined in the Python schema.

```javascript
// Locate the hidden backend widget
const dataWidget = node.widgets?.find(w => w.name === "my_json_data");
if (dataWidget) {
    dataWidget.type = "hidden"; // Hide the native ComfyUI text field
    dataWidget.computeSize = () => [0, -4];
    
    // 1. On Load: Deserialize JSON from the ComfyUI widget into Vue
    if (dataWidget.value) {
        instance.deserialise(dataWidget.value);
    }
    
    // 2. On Change: Serialize Vue state to JSON and store it in the ComfyUI widget
    // This should be triggered by the `onChange` callback passed to the Vue app
    dataWidget.value = instance.serialise();
}
```

### Global API Events (Progress Tracking)
Use the api object from ComfyUI to track execution progress or system states dynamically within your Vue components. Remember to remove listeners on cleanup.

```javascript
import { api } from "../../../scripts/api.js";

// Inside Vue setup() or onMounted():
function onProgressState(e) {
    const { nodes } = e.detail;
    // nodes is a Record<string, { value, max, state }>
    // Check if your node is "running" to update a progress bar
}

function onExecuting(e) {
    if (e.detail === null) {
        // Execution completely finished -> reset UI
    }
}

api.addEventListener("progress_state", onProgressState);
api.addEventListener("executing", onExecuting);

// CRITICAL: Remove listeners on unmount
function cleanup() {
    api.removeEventListener("progress_state", onProgressState);
    api.removeEventListener("executing", onExecuting);
}
```

### Minimum Node Dimensions Enforcement

```javascript
const MIN_W = 360, MIN_H = 180;
const origOnResize = node.onResize;
node.onResize = function(size) {
    size[0] = Math.max(MIN_W, size[0]);
    size[1] = Math.max(MIN_H, size[1]);
    origOnResize?.call(this, size);
};
```

### Extension Lifecycle Hooks (Execution Order)
When registering an extension via `comfyApp.registerExtension({...})`, the following hooks fire in this order. Only define the hooks you actually need.

| Hook | When It Fires | Typical Use |
|---|---|---|
| `init(app)` | Immediately on script parse, before LiteGraph canvas exists. | Inject global CSS, initialize shared state. |
| `addCustomNodeDefs(defs, app)` | Before node types are registered. | Programmatically inject frontend-only node definitions. |
| `getCustomWidgets(app)` | During widget registration phase. | Define new widget types beyond LiteGraph defaults. |
| `beforeRegisterNodeDef(nodeType, nodeData, app)` | Once per node **class** as it registers. | Globally modify a node class's appearance or default widgets before any instance is created. |
| `setup(app)` | After the app, UI, and canvas are fully loaded. | Bind global DOM event listeners, make initial API fetches. |
| `nodeCreated(node, app)` | Every time a user adds a node **instance** to the canvas. | Inject Vue DOM widgets, modify specific node instance behavior. |

### Context Menu Integration (Modern API)
To add custom right-click menu items, use declarative hooks — **never** monkey-patch `LGraphCanvas.prototype` or `nodeType.prototype`.

```javascript
comfyApp.registerExtension({
    name: "Duffy.MyNode.Vue",

    // Right-click on empty canvas background
    getCanvasMenuItems(canvas) {
        return [{ content: "My Custom Action", callback: () => { /* ... */ } }];
    },

    // Right-click on a specific node instance
    getNodeMenuItems(node) {
        if (node.comfyClass !== "Duffy_MyNode") return [];
        return [{ content: "Reset Parameters", callback: () => { /* ... */ } }];
    },
});
```

---

## 4. Server–Client Communication

### WebSocket: Server-to-Client Push
Use `PromptServer.instance.send_sync()` to broadcast real-time events (progress, status) from the Python backend to the JavaScript frontend.

**Python (backend):**
```python
from server import PromptServer

# Inside execute() or any server-side code:
PromptServer.instance.send_sync("duffy.my_event", {
    "node_type": "Duffy_MyNode",
    "progress": 0.75,
})
```

**JavaScript (frontend):**
```javascript
import { api } from "../../scripts/api.js";

api.addEventListener("duffy.my_event", (event) => {
    const { progress } = event.detail;
    // Update Vue component state, progress bar, etc.
});
// CRITICAL: Remove listener in cleanup()
```

### Custom REST Endpoints
Register custom HTTP routes on the Python backend using `@PromptServer.instance.routes`.

```python
from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get("/api/duffy/my_data")
async def get_my_data(request):
    # Validate inputs, avoid path traversal, return JSON
    return web.json_response({"status": "success", "items": []})
```

Fetch from the frontend:
```javascript
const response = await api.fetchApi("/api/duffy/my_data");
const data = await response.json();
```

---

## 5. Security Rules

* **DOM sanitization:** Always use `textContent` (not `innerHTML`) when rendering user-provided or workflow-provided strings into the DOM. Unsanitized `innerHTML` in a shared workflow JSON can escalate from XSS to RCE via custom backend endpoints.
* **Backend input validation:** All custom REST endpoint handlers must validate incoming payloads. Prevent path traversal when any endpoint accepts file paths — reject `..` segments and resolve against an allowed base directory.
* **No secrets in frontend code:** Never embed API keys, tokens, or credentials in JavaScript extensions.

---

## 6. Strict Anti-Patterns (Do NOT Generate)

* Do not generate inline vanilla JS DOM manipulation (`document.createElement`) for complex UI. Always use Vue 3 SFCs and Vite.

* Do not define `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`, or `CATEGORY` as class dictionaries or tuples.

* Do not register nodes by mutating `NODE_CLASS_MAPPINGS`.

* Do not use the legacy `IS_CHANGED` class method.

* Do not execute global DOM queries like `document.getElementById` expecting Vue elements to be fully mounted during early instantiation phases.

* Do not use dict-style `inputs={}` in `define_schema` — use list-style `inputs=[ ]`.

* Do not return raw tuples from `execute()` — always use `io.NodeOutput(...)`.

* Do not store state in `self` or `cls` — nodes must be completely stateless.

* Do not mutate input tensors or dicts directly — always `.clone()` or `.copy()` before modifying.

* Do not monkey-patch `LGraphCanvas.prototype` or `nodeType.prototype` to add context menu items. Use the declarative `getCanvasMenuItems` / `getNodeMenuItems` extension hooks instead.

* Do not use `innerHTML` to render user-provided or workflow-provided strings. Use `textContent` to prevent XSS.

---

## 7. Boilerplate Templates

### 1. Vite Configuration (`vite.config.ts`)

```typescript
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import cssInjectedByJs from "vite-plugin-css-injected-by-js";

export default defineConfig({
  plugins: [vue(), cssInjectedByJs({ topExecutionPriority: false })],
  build: {
    lib: { entry: "./src/main.ts", formats: ["es"], fileName: "my_node_widget" },
    rollupOptions: {
      external: ["../../scripts/app.js", "../../scripts/api.js"],
      output: { dir: "web", entryFileNames: "[name].js" },
    },
    minify: false,
  }
});
```

### 2. Vue Component (`src/MyWidget.vue`)

```vue
<template>
  <div class="my-widget-root">
    <h4>My Node Control</h4>
    <input type="range" v-model.number="myValue" @input="emitChange" />
    <span>{{ myValue }}</span>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import { api } from "../../scripts/api.js";

const props = defineProps<{ onChange?: (json: string) => void }>();
const myValue = ref(50);

function serialise() { return JSON.stringify({ value: myValue.value }); }
function deserialise(json: string) {
  try {
    const data = JSON.parse(json);
    if (data.value !== undefined) myValue.value = data.value;
  } catch (e) { /* ignore */ }
}

function emitChange() { props.onChange?.(serialise()); }

function cleanup() {
    // Remove global event listeners here
}

defineExpose({ serialise, deserialise, cleanup });
</script>

<style scoped>
.my-widget-root {
  padding: 8px;
  background: #222;
  color: #ddd;
  border-radius: 6px;
}
</style>
```

### 3. Vue Integration Entrypoint (`src/main.ts`)

```typescript
import { createApp } from "vue";
import { app as comfyApp } from "../../scripts/app.js";
import MyWidget from "./MyWidget.vue";

comfyApp.registerExtension({
    name: "Duffy.MyNode.Vue",

    async nodeCreated(node: any) {
        if (node.comfyClass !== "Duffy_MyNode") return;

        const dataWidget = node.widgets?.find((w: any) => w.name === "json_data");
        if (dataWidget) {
            dataWidget.type = "hidden";
            dataWidget.computeSize = () => [0, -4];
        }

        const container = document.createElement("div");
        container.style.cssText = "width:100%; box-sizing:border-box; overflow:hidden;";
        container.addEventListener("pointerdown", (e) => e.stopPropagation());
        container.addEventListener("wheel", (e) => e.stopPropagation());

        const vueApp = createApp(MyWidget, {
            onChange: (json: string) => {
                if (dataWidget) dataWidget.value = json;
                node.setDirtyCanvas(true, true);
            }
        });

        const instance = vueApp.mount(container) as any;

        const domWidget = node.addDOMWidget("vue_ui", "custom", container, { serialize: false });
        domWidget.computeSize = () => [300, 200];

        // Restore saved workflow state
        if (dataWidget?.value) instance.deserialise(dataWidget.value);

        // Safe unmount
        const origRemoved = node.onRemoved;
        node.onRemoved = function () {
            instance.cleanup?.();
            vueApp.unmount();
            origRemoved?.apply(this, arguments);
        };
    }
});
```

### 4. Python Backend (`nodes/my_node.py`)

```python
import torch
import json
from comfy_api.latest import io

class DuffyMyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Duffy_MyNode",
            display_name="My Node",
            category="Duffy/Example",
            description="Vue 3 powered custom node.",
            inputs=[
                io.String.Input("json_data", default="{}", socketless=True),
                io.Image.Input("image", display_name="Image"),
            ],
            outputs=[
                io.Image.Output("image", display_name="Image"),
            ],
        )

    @classmethod
    def execute(cls, json_data: str, image: torch.Tensor, **kwargs) -> io.NodeOutput:
        try:
            data = json.loads(json_data)
            val = data.get("value", 50)
        except json.JSONDecodeError:
            val = 50
            
        # Do something with val and image...
        return io.NodeOutput(image)
```
