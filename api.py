# api.py
# 【前后端接口层】
# 职责：接收前端请求 + 触发批量测评任务 + 提供全部统计数据查询接口
# 完全与攻击、评估业务逻辑解耦，只做接口转发、数据返回
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional

# 项目内部模块导入（兼容未安装为包时的本地文件导入）
_BASE_DIR = Path(__file__).resolve().parent


def _load_local_module(module_name: str, file_name: str) -> ModuleType:
    module_path = _BASE_DIR / file_name
    if not module_path.exists():
        raise ImportError(f"Local module file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_evaluator_module = _load_local_module("evaluator", "evaluator.py")
calculate_full_evaluation_metrics = _evaluator_module.calculate_full_evaluation_metrics

try:
    _runner_module = _load_local_module("attack_runner", "attack_runner.py")
    batch_run_scene_test = _runner_module.batch_run_scene_test
except ImportError:
    async def batch_run_scene_test(*args, **kwargs):
        raise RuntimeError("未找到 attack_runner.py，无法执行 /api/batch/start-test 批量测评接口。")

# ====================== FastAPI服务初始化 ======================
app = FastAPI(title="大模型红队安全评估系统API", version="1.0")

# 全局跨域配置（前端访问必备，全部放行）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== 前端传入参数模型（表单请求体） ======================
class BatchTestRequest(BaseModel):
    """
    前端表单传入参数：完全对应学长需求
    前端只需要传入：被测模型配置、选择测试的A1~A5场景、选择攻击变异方法
    所有测试用例后端自动加载，前端无需传入任何prompt
    """
    model_name: str
    model_api_key: str
    model_endpoint: str
    scene_list: list[str]       # 例：["A1","A2"]
    attack_method_list: list[str] # 例：["base_variation","word_replace"]

# ====================== 全部API接口定义（完整覆盖全部需求） ======================
# 1. 根路径健康检查
@app.get("/")
def root():
    return {
        "msg": "大创评估API服务运行正常",
        "接口文档地址": "/docs",
        "全部接口功能": {
            "批量测评触发": "/api/batch/start-test",
            "全局总体统计": "/api/stat/overall",
            "模型场景统计(A1~A5)": "/api/stat/model-scene",
            "模型攻击方法统计": "/api/stat/model-method",
            "攻击方法全局评估(含方差)": "/api/stat/attack-method",
            "全部原始测试明细日志": "/api/stat/detail-log"
        }
    }

# 2. 【核心接口】前端表单触发批量全场景测评任务
@app.post("/api/batch/start-test", summary="前端触发批量测评任务")
async def api_start_batch_test(request: BatchTestRequest):
    """
    前端传入模型、场景、攻击方法参数，后端全自动完成全流程：
    加载本地数据集 → 真实PyRIT攻击 → 全字段统计 → 分级判定 → 全部数据入库
    """
    # 组装被测模型配置
    model_config = {
        "model_name": request.model_name,
        "api_key": request.model_api_key,
        "endpoint": request.model_endpoint
    }
    # 执行批量完整测评
    result = await batch_run_scene_test(
        model_config=model_config,
        scene_list=request.scene_list,
        attack_method_list=request.attack_method_list
    )
    return {
        "code": 200,
        "status": "success",
        "msg": "批量测评任务全部执行完成，所有数据已入库",
        "data": result
    }

# 3. 全局总体统计查询接口
@app.get("/api/stat/overall", summary="全局总体评估统计")
def get_overall_stat():
    all_data = calculate_full_evaluation_metrics()
    return {"code":200, "data": all_data["overall_stat"]}

# 4. 模型×A1~A5场景统计查询接口（老师核心需求）
@app.get("/api/stat/model-scene", summary="模型各场景维度统计")
def get_model_scene_stat():
    all_data = calculate_full_evaluation_metrics()
    return {"code":200, "data": all_data["model_scene_stat"]}

# 5. 模型×攻击变异方法统计查询接口
@app.get("/api/stat/model-method", summary="模型各攻击方法维度统计")
def get_model_method_stat():
    all_data = calculate_full_evaluation_metrics()
    return {"code":200, "data": all_data["model_method_stat"]}

# 6. 攻击方法全局评估（含成功率方差）接口
@app.get("/api/stat/attack-method", summary="攻击方法全维度评估（含成功率方差）")
def get_attack_method_stat():
    all_data = calculate_full_evaluation_metrics()
    return {
        "code":200,
        "data": {
            "基础指标统计": all_data["attack_method_base"],
            "成功率方差(稳定性)": all_data["attack_method_variance"]
        }
    }

# 7. 全部原始明细日志查询接口（前端详情表格展示）
@app.get("/api/stat/detail-log", summary="所有测试任务原始明细日志")
def get_all_detail_log(keyword: Optional[str] = None):
    all_data = calculate_full_evaluation_metrics()
    logs = all_data["all_detail_log"]

    if keyword and keyword.strip():
        keyword_lower = keyword.strip().lower()
        filter_fields = (
            "task_id",
            "model_name",
            "scene_id",
            "attack_method",
            "original_prompt",
            "final_varied_prompt",
            "model_response",
        )
        logs = [
            row
            for row in logs
            if any(keyword_lower in str(row.get(field, "")).lower() for field in filter_fields)
        ]

    return {"code": 200, "keyword": keyword, "total_count": len(logs), "data": logs}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)