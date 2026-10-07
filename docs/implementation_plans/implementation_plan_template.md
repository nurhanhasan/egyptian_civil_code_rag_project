# Implementation Plan: `search_by_ingredient_ids()`

## 1. Goal

Implement ... to ...


## 2. API Contract (Example)

| Endpoint | Method | Description | Request Schema / Query Params  | Success Response | Failure Responses | Comment |
|---|---|---|---|---|---|---|
| `/api/v1/recipes/search?ingredient_ids=` | GET | Finds recipe IDs based on a list of ingredient IDs. | Query params (repeated): <br /> `ingredient_ids` => int <br /> `lang` => str | `200, list[RecipeListItem]` | `422, ErrorValidation` | 422: <br> Invalid `lang` value handled by get_lang dependency. <br> Invalid `ingredient_ids` automatically raises by fastAPI. |


## 3. Target Architecture (Example)

``` text
Recipes Router
      ↓
RecipeService.search_by_ingredient_ids()
      ↓
RecipeRepository.find_recipe_ids_by_ingredient_ids()
      ↓
Recipes__Ings
      ↓
PostgreSQL
      ↓
(ing_id, recipe_id) index
```

## 4. Responsibilities


## 5. Tests

### Repository tests

Test:

-   

### Service tests

Verify:

-   

### API tests

Test:

``` text
GET /
```

Also test malformed IDs according to the API contract.


## 6. Deliverables

-   

## 7. Acceptance Criteria

-   [ ] 
