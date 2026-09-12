# FMN LLM Layer Integration

Copy `src/llm/` into the current FMN submission repository.

Do not replace the current forecasting, risk, assessment, or configuration wholesale.

## Configuration

Merge only the `llm:` block from the experimental project into the current `config.yaml`.

Keep the current validated forecast configuration. The experimental config uses `ma28` and must not replace the current pooled LightGBM champion.

## Required dependencies

The LLM layer requires the `openai` package. YAML configuration continues to use the existing YAML dependency.

## Runtime contract

Analytics decide:
forecast → expected replenishment → projected inventory → risk → priority → drivers.

The LLM only:
1. explains a deterministic SKU assessment
2. interprets a planner question
3. calls deterministic structured tools
4. summarizes returned evidence

If no API key is configured or a provider fails, the core dashboard must remain functional and deterministic explanations should be used.

## Next integration step

Expose the LLM layer through FastAPI, then connect the Dash SKU Analysis and Ask the Data pages. Do not put dataframe calculations inside prompts.
