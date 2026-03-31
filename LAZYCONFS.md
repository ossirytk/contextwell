# LazyVim Python LSP Setup (Ruff + Pyrefly)

## Overview

This setup configures Neovim (LazyVim) for Python development using:
- **Ruff** — linting and formatting LSP
- **Pyrefly** — type-checking LSP (replaces Pyright/BasedPyright)

Both Pyright and BasedPyright are explicitly disabled to avoid conflicts.

## Prerequisites

Install the LSP servers (e.g. via Mason in Neovim, or manually):
- `ruff` — `pip install ruff` or `brew install ruff`
- `pyrefly` — `pip install pyrefly`

## Files to Create

### `~/.config/nvim/lua/config/options.lua`

Add to your `options.lua` to disable LazyVim's automatic Python LSP selection. This must be set early, before lazy.nvim starts:

```lua
-- Disable LazyVim's auto Python LSP selection
vim.g.lazyvim_python_lsp = "none"
vim.g.lazyvim_python_ruff = "none"
```

### `~/.config/nvim/lua/plugins/python.lua`

Configures Ruff and Pyrefly LSP servers, and disables Pyright/BasedPyright:

```lua
return {
  {
    "neovim/nvim-lspconfig",
    opts = function(_, opts)
      opts.servers = opts.servers or {}

      -- Disable Pyright / BasedPyright
      opts.servers.pyright = { enabled = false }
      opts.servers.basedpyright = { enabled = false }

      -- Ruff LSP
      opts.servers.ruff = {
        init_options = {
          settings = {
            logLevel = "error",
          },
        },
      }

      -- Pyrefly LSP
      -- nvim-lspconfig ships lsp/pyrefly.lua with cmd/filetypes/root_markers.
      -- on_exit suppresses the noisy code-0 exit notice.
      opts.servers.pyrefly = {
        on_exit = function(code, _, _)
          if code ~= 0 then
            vim.schedule(function()
              vim.notify("Pyrefly LSP exited with code: " .. code, vim.log.levels.WARN)
            end)
          end
        end,
        settings = {
          python = {
            pyrefly = {
              diagnosticMode = "workspace",
            },
          },
        },
      }

      return opts
    end,
  },
}
```

### `~/.config/nvim/lua/config/keymaps.lua`

Add to your `keymaps.lua` to use LSP go-to-definition on `Ctrl+LeftMouse`:

```lua
-- Ctrl+LeftMouse: use LSP go-to-definition instead of ctags
vim.keymap.set("n", "<C-LeftMouse>", "<LeftMouse><cmd>lua vim.lsp.buf.definition()<CR>", { desc = "Go to definition" })
```

### `~/.config/nvim/lua/config/autocmds.lua`

Add this to your existing `autocmds.lua` (or create it if it doesn't exist):

```lua
-- Disable Ruff hover (defer hover docs to Pyrefly instead)
-- Note: vim.lsp.get_client_by_id was removed in Neovim 0.12; use vim.lsp.get_clients instead
vim.api.nvim_create_autocmd("LspAttach", {
  group = vim.api.nvim_create_augroup("lsp_disable_ruff_hover", { clear = true }),
  callback = function(args)
    local client = vim.lsp.get_clients({ id = args.data.client_id })[1]
    if client and client.name == "ruff" then
      client.server_capabilities.hoverProvider = false
    end
  end,
})
```

## Notes

- `diagnosticMode = "workspace"` means Pyrefly checks all files in the project, not just open buffers.
- Ruff's `logLevel = "error"` suppresses noisy info/warning logs from the LSP server.
- LazyVim's `lazyvim_python_lsp = "none"` and `lazyvim_python_ruff = "none"` prevent LazyVim from auto-enabling its own Python extras on top of these.
- The `autocmds.lua` autocommand disables Ruff's hover provider on attach — this means `K` (hover) shows Pyrefly's type info instead of Ruff's. Without this, two LSPs compete for hover.
- Both plugin files go in `lua/plugins/` — LazyVim loads all `.lua` files in that directory automatically.
- Set `mason = false` on both ruff and pyrefly if they are managed outside Mason (e.g. via `uv tool install`). Mason requires `python3`/`pip` in PATH; if you use `uv` for Python, these are typically not available globally.
- **Neovim 0.12+**: `vim.lsp.get_client_by_id` was removed. Use `vim.lsp.get_clients({ id = ... })[1]` instead.
- **Neovim 0.12 + nvim-ts-autotag**: `vim.treesitter.get_parser()` now returns `nil` instead of raising when no parser is found. This causes `nvim-ts-autotag` to crash on `InsertLeave` in Python buffers. Add `~/.config/nvim/lua/plugins/autotag.lua` to patch it:

```lua
return {
  {
    "windwp/nvim-ts-autotag",
    config = function(_, opts)
      require("nvim-ts-autotag").setup(opts)
      local internal = require("nvim-ts-autotag.internal")
      local orig_rename = internal.rename_tag
      internal.rename_tag = function()
        local ok, parser = pcall(vim.treesitter.get_parser)
        if not ok or not parser then
          return
        end
        orig_rename()
      end
    end,
  },
}
```
