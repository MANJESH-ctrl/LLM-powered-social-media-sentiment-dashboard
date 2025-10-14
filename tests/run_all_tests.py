import sys
import os

def run_all_tests():
    """Run all tests and provide a comprehensive report"""
    print("🚀 RUNNING COMPREHENSIVE TEST SUITE")
    print("=" * 50)
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"📁 Project root: {project_root}")
    
    test_results = {}
    
    # Run dependency test
    from test_dependencies import test_dependencies
    test_results['dependencies'] = test_dependencies()
    
    print("\n" + "=" * 50)
    
    # Run config test
    from test_config import test_config
    test_results['config'] = test_config()
    
    print("\n" + "=" * 50)
    
    # Run imports test
    from test_imports import test_imports
    test_results['imports'] = test_imports()
    
    print("\n" + "=" * 50)
    
    # Run collectors test
    from test_collectors import test_collectors
    test_results['collectors'] = test_collectors()
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! You can run the app now.")
        print("💡 Run: uv run streamlit run src/dashboard/app.py")
    else:
        print("🚨 SOME TESTS FAILED! Please fix the issues above.")
        print("💡 Check the specific test outputs for details.")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()