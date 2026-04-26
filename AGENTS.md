# AGENT约束规则

## python环境

本项目使用pixi管理python环境，执行python命令前需要加上pixi run前缀。

## Architecture Contract

For any task that changes repository structure, Hydra configs, simulation entrypoints, asset paths, or module boundaries, first read and follow `docs/repo_architecture_contract.md`.

If a task is only a local logic fix and does not affect those areas, you may proceed directly, but do not introduce changes that conflict with the contract.