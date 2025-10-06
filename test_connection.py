"""
Database Connection Test for EWS2
Run this script to test your SQL Server connection
"""

def test_pyodbc():
    """Test if pyodbc is installed"""
    try:
        import pyodbc
        print("✅ pyodbc module is installed")
        
        # List available drivers
        drivers = [x for x in pyodbc.drivers() if 'SQL Server' in x]
        if drivers:
            print(f"✅ SQL Server ODBC drivers found: {drivers}")
        else:
            print("⚠️  No SQL Server ODBC drivers found")
            print("   Install 'ODBC Driver 17 for SQL Server' from Microsoft")
        return True
    except ImportError:
        print("❌ pyodbc module not installed")
        print("   Install with: pip install pyodbc")
        return False

def test_database_connection():
    """Test database connection"""
    try:
        from ews2_database import EWS2Database
        print("✅ EWS2Database module imported successfully")
        
        # Connection details
        print("\n🔄 Testing database connection...")
        print("   Server: localhost")
        print("   Database: EWS2_StudentSuccess")
        print("   Authentication: Windows Authentication")
        
        # Test connection
        db = EWS2Database()
        
        if db.test_connection():
            print("✅ Database connection successful!")
            
            # Test basic operations
            try:
                student_count = db.get_student_count()
                print(f"📊 Students in database: {student_count}")
                
                if student_count > 0:
                    print("✅ Database contains student data")
                    
                    # Show sample data
                    students = db.get_all_students()
                    if not students.empty:
                        print(f"📋 Sample student IDs: {list(students['student_id'].head(3))}")
                else:
                    print("⚠️  Database is empty")
                    print("   Use: python ews2_main.py --demo")
                    
            except Exception as e:
                print(f"⚠️  Database connected but error accessing data: {e}")
                
        else:
            print("❌ Database connection failed!")
            print_troubleshooting()
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print_troubleshooting()

def print_troubleshooting():
    """Print troubleshooting steps"""
    print("\n🔧 Troubleshooting Steps:")
    print("1. Verify SQL Server is running")
    print("2. Check if database 'EWS2_StudentSuccess' exists")
    print("3. Run database_setup.sql in SQL Server Management Studio")
    print("4. Ensure Windows Authentication is enabled")
    print("5. Install ODBC Driver 17 for SQL Server if missing")

def main():
    print("="*60)
    print("  EWS2 Database Connection Test")
    print("="*60)
    
    # Test pyodbc installation
    if not test_pyodbc():
        return
    
    # Test database connection
    test_database_connection()
    
    print("\n" + "="*60)
    print("  Test Complete")
    print("="*60)

if __name__ == "__main__":
    main()
