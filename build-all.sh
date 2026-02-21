#!/bin/bash
# =============================================================================
# Build and Test All Docker Targets
# =============================================================================
# Usage:
#   ./build-all.sh              # Build and test all targets
#   ./build-all.sh --build-only # Build without testing
#   ./build-all.sh --test-only  # Test existing images only
#   ./build-all.sh base ml      # Build and test specific targets
# =============================================================================

set -e

# Configuration
IMAGE_PREFIX="ds"
DOCKERFILE="Dockerfile"

# All available targets in dependency order
ALL_TARGETS=(
    "base"
    "scientific"
    "visualization"
    "dataio"
    "ml"
    "deeplearn"
    "vision"
    "audio"
    "geospatial"
    "timeseries"
    "nlp"
    "full"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track results using parallel arrays (bash 3.x compatible)
BUILD_RESULT_TARGETS=()
BUILD_RESULT_VALUES=()
TEST_RESULT_TARGETS=()
TEST_RESULT_VALUES=()

# Helper functions to simulate associative arrays
set_build_result() {
    local target=$1
    local value=$2
    BUILD_RESULT_TARGETS+=("$target")
    BUILD_RESULT_VALUES+=("$value")
}

get_build_result() {
    local target=$1
    local i
    for i in "${!BUILD_RESULT_TARGETS[@]}"; do
        if [ "${BUILD_RESULT_TARGETS[$i]}" = "$target" ]; then
            echo "${BUILD_RESULT_VALUES[$i]}"
            return
        fi
    done
    echo "skipped"
}

set_test_result() {
    local target=$1
    local value=$2
    TEST_RESULT_TARGETS+=("$target")
    TEST_RESULT_VALUES+=("$value")
}

get_test_result() {
    local target=$1
    local i
    for i in "${!TEST_RESULT_TARGETS[@]}"; do
        if [ "${TEST_RESULT_TARGETS[$i]}" = "$target" ]; then
            echo "${TEST_RESULT_VALUES[$i]}"
            return
        fi
    done
    echo "skipped"
}

print_header() {
    echo ""
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

build_target() {
    local target=$1
    local image_name="${IMAGE_PREFIX}-${target}"

    echo ""
    echo -e "${YELLOW}Building target: ${target}${NC}"
    echo "  Image: ${image_name}"
    echo "  Command: docker build --target ${target} -t ${image_name} ."
    echo ""

    if docker build --target "${target}" -t "${image_name}" -f "${DOCKERFILE}" .; then
        set_build_result "$target" "success"
        print_success "Built ${image_name}"
        return 0
    else
        set_build_result "$target" "failed"
        print_error "Failed to build ${image_name}"
        return 1
    fi
}

test_target() {
    local target=$1
    local image_name="${IMAGE_PREFIX}-${target}"

    echo ""
    echo -e "${YELLOW}Testing target: ${target}${NC}"

    # Check if image exists
    if ! docker image inspect "${image_name}" &> /dev/null; then
        print_error "Image ${image_name} not found. Build it first."
        set_test_result "$target" "not_found"
        return 1
    fi

    # --- Step 1: import verification (fast smoke test) ---
    local verify_script
    if [ "${target}" = "full" ] || [ "${target}" = "base" ]; then
        verify_script="/home/jupyter/scripts/verify_imports.py"
    else
        verify_script="/home/jupyter/scripts/verify_${target}.py"
    fi

    echo "  [1/2] Import verification"
    echo "  Running: docker run --rm ${image_name} uv run --no-project python ${verify_script}"
    echo ""

    if ! docker run --rm "${image_name}" uv run --no-project python "${verify_script}"; then
        set_test_result "$target" "failed"
        print_error "Import verification failed for ${image_name}"
        return 1
    fi

    # --- Step 2: example smoke tests (run actual example scripts) ---
    # base has no example tests; skip pytest for it.
    if [ "${target}" = "base" ]; then
        set_test_result "$target" "success"
        print_success "Tests passed for ${image_name} (no examples for base target)"
        return 0
    fi

    # Build pytest -m expression: full runs all marks, others run their own mark.
    local pytest_marks
    if [ "${target}" = "full" ]; then
        # Run every marked test (skip slow/network-dependent by default)
        pytest_marks="scientific or visualization or dataio or ml or deeplearn or vision or audio or geospatial or timeseries or nlp"
    else
        pytest_marks="${target}"
    fi

    echo ""
    echo "  [2/2] Example smoke tests"
    echo "  Running: docker run --rm ${image_name} uv run --no-project python -m pytest /home/jupyter/tests/ -m \"${pytest_marks}\" -v --timeout=300"
    echo ""

    # Exit code 5 means "no tests collected" — treat as success (mark not yet populated).
    local pytest_exit
    docker run --rm "${image_name}" \
        uv run --no-project python -m pytest /home/jupyter/tests/ \
        -m "${pytest_marks}" \
        -v \
        --timeout=300
    pytest_exit=$?

    if [ "${pytest_exit}" -eq 0 ] || [ "${pytest_exit}" -eq 5 ]; then
        set_test_result "$target" "success"
        print_success "Tests passed for ${image_name}"
        return 0
    else
        set_test_result "$target" "failed"
        print_error "Example tests failed for ${image_name} (exit code ${pytest_exit})"
        return 1
    fi
}

get_image_size() {
    local image_name=$1
    docker image inspect "${image_name}" --format='{{.Size}}' 2>/dev/null | awk '{printf "%.2f GB", $1/1024/1024/1024}'
}

print_summary() {
    print_header "BUILD & TEST SUMMARY"

    echo ""
    echo "Build Results:"
    echo "--------------"
    for target in "${TARGETS[@]}"; do
        local result
        result=$(get_build_result "$target")
        local image_name="${IMAGE_PREFIX}-${target}"
        local size=""

        if [ "$result" = "success" ]; then
            size=" ($(get_image_size ${image_name}))"
            print_success "${target}${size}"
        elif [ "$result" = "failed" ]; then
            print_error "${target}"
        else
            echo "  - ${target} (skipped)"
        fi
    done

    if [ "$RUN_TESTS" = true ]; then
        echo ""
        echo "Test Results:"
        echo "-------------"
        for target in "${TARGETS[@]}"; do
            local result
            result=$(get_test_result "$target")
            if [ "$result" = "success" ]; then
                print_success "${target}"
            elif [ "$result" = "failed" ]; then
                print_error "${target}"
            elif [ "$result" = "not_found" ]; then
                print_warning "${target} (image not found)"
            else
                echo "  - ${target} (skipped)"
            fi
        done
    fi

    echo ""

    # Count failures
    local build_failures=0
    local test_failures=0

    for target in "${TARGETS[@]}"; do
        local build_res test_res
        build_res=$(get_build_result "$target")
        test_res=$(get_test_result "$target")
        [ "$build_res" = "failed" ] && build_failures=$((build_failures + 1))
        [ "$test_res" = "failed" ] && test_failures=$((test_failures + 1))
    done

    if [ $build_failures -eq 0 ] && [ $test_failures -eq 0 ]; then
        print_success "All operations completed successfully!"
        return 0
    else
        [ $build_failures -gt 0 ] && print_error "${build_failures} build(s) failed"
        [ $test_failures -gt 0 ] && print_error "${test_failures} test(s) failed"
        return 1
    fi
}

show_usage() {
    echo "Usage: $0 [OPTIONS] [TARGETS...]"
    echo ""
    echo "Options:"
    echo "  --build-only    Build images without running tests"
    echo "  --test-only     Run tests on existing images only"
    echo "  --help          Show this help message"
    echo ""
    echo "Available targets:"
    for target in "${ALL_TARGETS[@]}"; do
        echo "  - ${target}"
    done
    echo ""
    echo "Examples:"
    echo "  $0                    # Build and test all targets"
    echo "  $0 --build-only       # Build all targets without testing"
    echo "  $0 base scientific    # Build and test only base and scientific"
    echo "  $0 --test-only full   # Test only the full image"
}

# Parse arguments
RUN_BUILD=true
RUN_TESTS=true
TARGETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            RUN_TESTS=false
            shift
            ;;
        --test-only)
            RUN_BUILD=false
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            # Check if it's a valid target
            valid=false
            for t in "${ALL_TARGETS[@]}"; do
                if [ "$1" = "$t" ]; then
                    valid=true
                    break
                fi
            done
            if [ "$valid" = true ]; then
                TARGETS+=("$1")
            else
                echo "Unknown option or target: $1"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# If no targets specified, use all
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=("${ALL_TARGETS[@]}")
fi

# Main execution
print_header "Data Science Docker Build System"
echo "Targets: ${TARGETS[*]}"
echo "Build: ${RUN_BUILD}"
echo "Test: ${RUN_TESTS}"

# Build phase
if [ "$RUN_BUILD" = true ]; then
    print_header "BUILDING IMAGES"
    for target in "${TARGETS[@]}"; do
        build_target "${target}" || true
    done
fi

# Test phase
if [ "$RUN_TESTS" = true ]; then
    print_header "TESTING IMAGES"
    for target in "${TARGETS[@]}"; do
        test_target "${target}" || true
    done
fi

# Summary
print_summary
