# [HTTP_METHOD] [ENDPOINT_URL]

[Brief description of what this endpoint does]

## Authentication

[State authentication requirements: "No authentication required" OR describe auth method: API key, OAuth, Bearer token, etc.]

## Parameters

### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `[param_name]` | [type] | [Yes/No] | [Description including constraints, format requirements, normalization behavior] |

### Request Headers

[Include this section only if headers affect behavior]

| Header | Required | Description |
|--------|----------|-------------|
| `[Header-Name]` | [Yes/No] | [Accepted values and behavior] |

[Optional: Add format precedence rules, special header behaviors, or notes about ignored parameters]

## Response Codes

| Code | Description |
|------|-------------|
| 200 OK | [Describe success conditions] |
| 404 Not Found | [Describe what triggers this] |
| [Other codes] | [When they occur] |

[Optional: Add notes about distinguishing between different error types if applicable]

[Optional: Include caching behavior if discovered: **Caching:** Responses are cached for [duration] (`Cache-Control: [value]`).]

## Response Structure

[Describe the structure hierarchy, e.g.: `parent` → `child[]` → `properties[]`]

[Optional: If success/failure detection is non-obvious, add: **Detecting [condition]:** [Explain how to determine state from response]]

**[Response Object] Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `[field_name]` | [type] | [Description, including typical values or behaviors] |

[Optional: List enum values if applicable: **[Field name] values:** value1, value2, value3, ...]
