# Refactoring Plan: mm_control and mm_plan

## Current State Analysis

### mm_control (2815 lines)
- **MPC.py**: Main controller class (539 lines) - contains hardcoded cost/constraint selection
- **MPCCostFunctions.py**: 17+ cost function classes with significant duplication
- **MPCConstraints.py**: Constraint classes (relatively clean)
- **robot.py**: Robot model interface (633 lines) - mixed concerns
- **map.py**: SDF map classes

### mm_plan (1852 lines)
- **TaskManager.py**: 6+ Stack of Tasks implementations with duplication
- **BasePlanner.py**: 7 base planner classes
- **EEPlanner.py**: 9 end-effector planner classes
- **PlanBaseClass.py**: Base classes (relatively clean)

## Key Issues Identified

1. **Code Duplication**: Many similar classes with only minor differences
2. **Hard-coded Logic**: Cost/constraint selection in MPC.__init__ uses if/else
3. **String-based Registration**: Planners registered via string lookup (fragile)
4. **Mixed Concerns**: Robot model, constraints, costs all intertwined
5. **Large Files**: Single files with too many responsibilities

## Refactoring Strategy: Start Small, Build Up

### Phase 1: Quick Wins (Low Risk, High Impact)

#### 1.1 Create Cost Function Registry
**Goal**: Replace hardcoded cost selection with registry pattern
**Risk**: Low - just adds abstraction layer
**Steps**:
1. Create `CostFunctionRegistry` class
2. Register all existing cost functions
3. Update MPC to use registry instead of hardcoded lists
4. Maintain backward compatibility

#### 1.2 Create Planner Registry
**Goal**: Replace string-based lookup with explicit registry
**Risk**: Low - improves type safety
**Steps**:
1. Create `PlannerRegistry` class
2. Register all existing planners
3. Update TaskManager to use registry
4. Add validation for unknown planner types

#### 1.3 Extract Cost Selection Logic
**Goal**: Move hardcoded if/else to configuration
**Risk**: Medium - changes behavior
**Steps**:
1. Create `CostSelector` class
2. Move logic from MPC.__init__ to selector
3. Make it configurable via YAML
4. Test thoroughly

### Phase 2: Consolidation (Medium Risk)

#### 2.1 Unify Similar Cost Functions
**Goal**: Reduce 17+ classes to ~5-7 core classes
**Risk**: Medium - requires careful testing
**Approach**:
- Identify patterns: Position, Velocity, Pose, Effort
- Create parameterized versions
- Keep old classes as thin wrappers initially
- Gradually migrate

#### 2.2 Consolidate Task Managers
**Goal**: Single configurable TaskManager instead of 6+ classes
**Risk**: Medium - core functionality
**Approach**:
- Use Strategy pattern
- Single TaskManager with strategy selection
- Keep old classes as strategies initially

### Phase 3: Architecture Improvements (Higher Risk)

#### 3.1 Split Large Files
**Goal**: Break down MPC.py and robot.py
**Risk**: Medium - many dependencies
**Approach**:
- Extract MPC construction logic
- Separate robot model from interface
- Create builder pattern for MPC

#### 3.2 Improve Separation of Concerns
**Goal**: Clear boundaries between modules
**Risk**: High - touches everything
**Approach**:
- Define clear interfaces
- Reduce coupling
- Improve dependency injection

## Implementation Plan: Start Here

### Step 1: Cost Function Registry (Easiest First)

Create a registry that maps cost function names to classes:

```python
# mm_control/src/mm_control/cost_registry.py
class CostFunctionRegistry:
    _registry = {}

    @classmethod
    def register(cls, name, cost_class):
        cls._registry[name] = cost_class

    @classmethod
    def create(cls, name, robot_model, params):
        if name not in cls._registry:
            raise ValueError(f"Unknown cost function: {name}")
        return cls._registry[name](robot_model, params)

    @classmethod
    def list_available(cls):
        return list(cls._registry.keys())
```

### Step 2: Planner Registry (Similar Pattern)

```python
# mm_plan/src/mm_plan/planner_registry.py
class PlannerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name, planner_class):
        cls._registry[name] = planner_class

    @classmethod
    def create(cls, config):
        planner_type = config["planner_type"]
        if planner_type not in cls._registry:
            raise ValueError(f"Unknown planner type: {planner_type}")
        return cls._registry[planner_type](config)
```

### Step 3: Configuration-Driven Cost Selection

Instead of:
```python
if config["base_pose_tracking_enabled"]:
    costs = [self.BasePoseSE2Cost, ...]
else:
    costs = [self.BasePos2Cost, ...]
```

Use:
```yaml
controller:
  costs:
    - type: BasePoseSE2
      enabled: true
      params: {...}
    - type: EEPos3
      enabled: true
      params: {...}
```

## Recommended Starting Point

**Start with Step 1 (Cost Function Registry)** because:
1. ✅ Low risk - just adds abstraction
2. ✅ High impact - enables future improvements
3. ✅ Easy to test - doesn't change behavior
4. ✅ Backward compatible - old code still works
5. ✅ Sets pattern for other refactorings

## Success Metrics

- [ ] All existing tests pass
- [ ] No performance regression
- [ ] Code is easier to understand
- [ ] Easier to add new cost functions/planners
- [ ] Configuration-driven instead of hardcoded
- [ ] Reduced code duplication

## Next Steps

1. **Create the registries** (Step 1 & 2)
2. **Update existing code to use them** (gradually)
3. **Add tests** for registry functionality
4. **Document** the new patterns
5. **Iterate** on consolidation (Phase 2)

## Risk Mitigation

- ✅ Start with registries (low risk, high value)
- ✅ Keep old code working (backward compatible)
- ✅ Test after each change
- ✅ Document as we go
- ✅ Small, focused commits
