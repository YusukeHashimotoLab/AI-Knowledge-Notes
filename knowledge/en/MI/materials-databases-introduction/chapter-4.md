---
title: "Chapter 4: Building Custom Databases"
chapter_title: "Chapter 4: Building Custom Databases"
subtitle: From SQLite to PostgreSQL - Structuring and Publishing Experimental Data
reading_time: 15-20 min
difficulty: Beginner to Intermediate
code_examples: 10
exercises: 3
version: 1.0
created_at: 2025-10-17
---

# Chapter 4: Building Custom Databases

This chapter shows the path from starting small with a local database to sharing and publishing data. We also cover fundamental operational rules and security considerations.

**💡 Note:** Clearly define roles (registration, approval, viewing). Decide on backup and access control from the start.

**From SQLite to PostgreSQL - Structuring and Publishing Experimental Data**

## Learning Objectives

By completing this chapter, you will be able to:

  * ✅ Design schemas for materials databases
  * ✅ Build practical databases with SQLite
  * ✅ Manage large-scale data with PostgreSQL
  * ✅ Implement backup and version control
  * ✅ Publish data and obtain DOIs

**Reading time** : 15-20 min **Code examples** : 10 **Exercises** : 3

* * *

## 4.1 Database Design Fundamentals

### 4.1.1 Schema Design Principles

In materials database schema design, we define the following three main tables:

  1. **Materials** : Basic material information
  2. **Properties** : Measured property data
  3. **Experiments** : Experimental conditions and metadata

**ER Diagram (Entity-Relationship Diagram)** :
    
    
    ```mermaid
    erDiagram
        Materials ||--o{ Properties : has
        Materials ||--o{ Experiments : tested_in
        Experiments ||--o{ Properties : produces
    
        Materials {
            int material_id PK
            string formula
            string structure_type
            float density
            datetime created_at
        }
    
        Properties {
            int property_id PK
            int material_id FK
            int experiment_id FK
            string property_type
            float value
            float uncertainty
            string unit
        }
    
        Experiments {
            int experiment_id PK
            string experiment_type
            datetime date
            string operator
            string conditions
        }
    ```

### 4.1.2 Normalization

**First Normal Form (1NF)** : Each column contains atomic values **Second Normal Form (2NF)** : Eliminate partial functional dependencies **Third Normal Form (3NF)** : Eliminate transitive functional dependencies

**Code Example 1: Schema Definition (SQLite)**
    
    
    import sqlite3
    
    # Database connection
    conn = sqlite3.connect('materials.db')
    cursor = conn.cursor()
    
    # Materials table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS materials (
        material_id INTEGER PRIMARY KEY AUTOINCREMENT,
        formula TEXT NOT NULL,
        structure_type TEXT,
        density REAL,
        space_group INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    )
    ''')
    
    # Properties table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS properties (
        property_id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_id INTEGER NOT NULL,
        experiment_id INTEGER,
        property_type TEXT NOT NULL,
        value REAL NOT NULL,
        uncertainty REAL,
        unit TEXT,
        measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (material_id) REFERENCES materials(material_id),
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
    )
    ''')
    
    # Experiments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_type TEXT NOT NULL,
        date DATE,
        operator TEXT,
        temperature REAL,
        pressure REAL,
        conditions TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create indexes (for faster queries)
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_formula
    ON materials(formula)
    ''')
    
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_property_type
    ON properties(property_type)
    ''')
    
    conn.commit()
    print("Database schema created successfully")
    

* * *

## 4.2 Local Database with SQLite

### 4.2.1 CRUD Operations

**Code Example 2: Create (Data Insertion)**
    
    
    import sqlite3
    from datetime import datetime
    
    conn = sqlite3.connect('materials.db')
    cursor = conn.cursor()
    
    def insert_material(formula, structure_type, density):
        """Insert material data"""
        cursor.execute('''
        INSERT INTO materials (formula, structure_type, density)
        VALUES (?, ?, ?)
        ''', (formula, structure_type, density))
    
        conn.commit()
        material_id = cursor.lastrowid
        print(f"Material added: ID={material_id}, {formula}")
        return material_id
    
    def insert_property(
        material_id,
        property_type,
        value,
        uncertainty=None,
        unit=None
    ):
        """Insert property data"""
        cursor.execute('''
        INSERT INTO properties
        (material_id, property_type, value, uncertainty, unit)
        VALUES (?, ?, ?, ?, ?)
        ''', (material_id, property_type, value, uncertainty, unit))
    
        conn.commit()
        print(f"Property added: {property_type}={value} {unit}")
    
    # Usage example
    mat_id = insert_material("TiO2", "rutile", 4.23)
    insert_property(mat_id, "band_gap", 3.0, 0.1, "eV")
    insert_property(mat_id, "refractive_index", 2.61, 0.02, "")
    

**Code Example 3: Read (Data Retrieval)**
    
    
    def query_materials(formula_pattern=None):
        """Search materials"""
        if formula_pattern:
            cursor.execute('''
            SELECT * FROM materials
            WHERE formula LIKE ?
            ''', (f"%{formula_pattern}%",))
        else:
            cursor.execute('SELECT * FROM materials')
    
        results = cursor.fetchall()
    
        print(f"Search results: {len(results)} items")
        for row in results:
            print(f"ID: {row[0]}, Formula: {row[1]}, "
                  f"Type: {row[2]}, Density: {row[3]}")
    
        return results
    
    def query_properties(material_id):
        """Retrieve property data"""
        cursor.execute('''
        SELECT p.property_type, p.value, p.uncertainty, p.unit
        FROM properties p
        WHERE p.material_id = ?
        ''', (material_id,))
    
        results = cursor.fetchall()
    
        print(f"\nProperties for material ID {material_id}:")
        for row in results:
            prop_type, value, unc, unit = row
            if unc:
                print(f"- {prop_type}: {value} ± {unc} {unit}")
            else:
                print(f"- {prop_type}: {value} {unit}")
    
        return results
    
    # Usage example
    query_materials("TiO2")
    query_properties(1)
    

**Code Example 4: Update (Data Modification)**
    
    
    def update_material(material_id, **kwargs):
        """Update material data"""
        set_clause = ", ".join(
            [f"{key} = ?" for key in kwargs.keys()]
        )
        values = list(kwargs.values()) + [material_id]
    
        cursor.execute(f'''
        UPDATE materials
        SET {set_clause}
        WHERE material_id = ?
        ''', values)
    
        conn.commit()
        print(f"Material ID {material_id} updated")
    
    # Usage example
    update_material(1, density=4.24, notes="Updated density")
    

**Code Example 5: Delete (Data Deletion)**
    
    
    def delete_material(material_id):
        """Delete material data (cascade)"""
        # Delete associated property data
        cursor.execute('''
        DELETE FROM properties WHERE material_id = ?
        ''', (material_id,))
    
        cursor.execute('''
        DELETE FROM materials WHERE material_id = ?
        ''', (material_id,))
    
        conn.commit()
        print(f"Material ID {material_id} deleted")
    
    # Usage example (use with caution)
    # delete_material(1)
    

* * *

## 4.3 Large-Scale Data Management with PostgreSQL/MySQL

### 4.3.1 PostgreSQL Connection

**Code Example 6: PostgreSQL Connection and Schema Creation**
    
    
    import psycopg2
    from psycopg2 import sql
    
    # PostgreSQL connection
    conn = psycopg2.connect(
        host="localhost",
        database="materials_db",
        user="your_username",
        password="your_password"
    )
    
    cursor = conn.cursor()
    
    # Create tables (PostgreSQL version)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS materials (
        material_id SERIAL PRIMARY KEY,
        formula VARCHAR(100) NOT NULL,
        structure_type VARCHAR(50),
        density REAL,
        space_group INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        notes TEXT,
        UNIQUE(formula)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS properties (
        property_id SERIAL PRIMARY KEY,
        material_id INTEGER NOT NULL,
        experiment_id INTEGER,
        property_type VARCHAR(50) NOT NULL,
        value REAL NOT NULL,
        uncertainty REAL,
        unit VARCHAR(20),
        measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (material_id)
            REFERENCES materials(material_id)
            ON DELETE CASCADE,
        FOREIGN KEY (experiment_id)
            REFERENCES experiments(experiment_id)
            ON DELETE SET NULL
    )
    ''')
    
    # GIN index (for full-text search)
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_formula_gin
    ON materials USING GIN (to_tsvector('english', formula))
    ''')
    
    conn.commit()
    print("PostgreSQL schema created successfully")
    

### 4.3.2 Bulk Insert

**Code Example 7: Efficient Bulk Data Insertion**
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    import pandas as pd
    from psycopg2.extras import execute_batch
    
    def bulk_insert_materials(df):
        """Bulk insert from DataFrame"""
        data = [
            (
                row['formula'],
                row['structure_type'],
                row['density']
            )
            for _, row in df.iterrows()
        ]
    
        insert_query = '''
        INSERT INTO materials (formula, structure_type, density)
        VALUES (%s, %s, %s)
        ON CONFLICT (formula) DO NOTHING
        '''
    
        execute_batch(cursor, insert_query, data, page_size=1000)
        conn.commit()
    
        print(f"{len(data)} records inserted")
    
    # Usage example
    df = pd.DataFrame({
        'formula': ['TiO2', 'ZnO', 'GaN', 'SiC'],
        'structure_type': ['rutile', 'wurtzite', 'wurtzite', 'zincblende'],
        'density': [4.23, 5.61, 6.15, 3.21]
    })
    
    bulk_insert_materials(df)
    

* * *

## 4.4 Backup Strategies

### 4.4.1 Periodic Backups

**Code Example 8: Automated Backup Script**
    
    
    import sqlite3
    import shutil
    from datetime import datetime
    import os
    
    class DatabaseBackup:
        """Database backup management"""
    
        def __init__(self, db_path, backup_dir="backups"):
            self.db_path = db_path
            self.backup_dir = backup_dir
            os.makedirs(backup_dir, exist_ok=True)
    
        def create_backup(self):
            """Create backup"""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"materials_db_{timestamp}.db"
            backup_path = os.path.join(
                self.backup_dir, backup_name
            )
    
            # Copy database
            shutil.copy2(self.db_path, backup_path)
    
            # Compress (optional)
            import gzip
            with open(backup_path, 'rb') as f_in:
                with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
            os.remove(backup_path)  # Delete uncompressed version
    
            print(f"Backup created: {backup_path}.gz")
            return f"{backup_path}.gz"
    
        def restore_backup(self, backup_file):
            """Restore from backup"""
            import gzip
    
            # Decompress
            temp_db = "temp_restored.db"
            with gzip.open(backup_file, 'rb') as f_in:
                with open(temp_db, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
            # Backup current database
            self.create_backup()
    
            # Restore
            shutil.move(temp_db, self.db_path)
            print(f"Database restored from: {backup_file}")
    
        def list_backups(self):
            """List backups"""
            backups = sorted(
                [f for f in os.listdir(self.backup_dir)
                 if f.endswith('.gz')]
            )
    
            print("=== Backup List ===")
            for i, backup in enumerate(backups, 1):
                size = os.path.getsize(
                    os.path.join(self.backup_dir, backup)
                ) / 1024  # KB
                print(f"{i}. {backup} ({size:.1f} KB)")
    
            return backups
    
    # Usage example
    backup_mgr = DatabaseBackup('materials.db')
    backup_mgr.create_backup()
    backup_mgr.list_backups()
    

### 4.4.2 Version Control (Git LFS)

**Code Example 9: Data Version Control with Git LFS**
    
    
    # Git LFS setup
    git lfs install
    
    # Track large files
    git lfs track "*.db"
    git lfs track "*.db.gz"
    
    # Automatically added to .gitattributes
    cat .gitattributes
    # *.db filter=lfs diff=lfs merge=lfs -text
    # *.db.gz filter=lfs diff=lfs merge=lfs -text
    
    # Commit
    git add materials.db .gitattributes
    git commit -m "Add materials database"
    git push origin main
    

* * *

## 4.5 Data Publication and DOI Acquisition

### 4.5.1 Uploading to Zenodo

**Code Example 10: Publishing Data via Zenodo API**
    
    
    # Requirements:
    # - Python 3.9+
    # - requests>=2.31.0
    
    import requests
    import json
    
    ZENODO_TOKEN = "your_zenodo_token"
    ZENODO_URL = "https://zenodo.org/api/deposit/depositions"
    
    def create_zenodo_deposit(
        title,
        description,
        creators,
        keywords
    ):
        """Create Zenodo deposit"""
    
        headers = {
            "Content-Type": "application/json"
        }
        params = {"access_token": ZENODO_TOKEN}
    
        # Metadata
        data = {
            "metadata": {
                "title": title,
                "upload_type": "dataset",
                "description": description,
                "creators": creators,
                "keywords": keywords,
                "access_right": "open",
                "license": "cc-by-4.0"
            }
        }
    
        # Create deposit
        response = requests.post(
            ZENODO_URL,
            params=params,
            json=data,
            headers=headers
        )
    
        if response.status_code == 201:
            deposition = response.json()
            deposition_id = deposition['id']
            bucket_url = deposition['links']['bucket']
    
            print(f"Deposit created successfully: ID={deposition_id}")
            return deposition_id, bucket_url
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None, None
    
    def upload_file_to_zenodo(bucket_url, file_path):
        """Upload file"""
        params = {"access_token": ZENODO_TOKEN}
    
        with open(file_path, 'rb') as f:
            response = requests.put(
                f"{bucket_url}/{os.path.basename(file_path)}",
                params=params,
                data=f
            )
    
        if response.status_code == 200:
            print(f"File uploaded successfully: {file_path}")
            return True
        else:
            print(f"Upload error: {response.status_code}")
            return False
    
    def publish_zenodo_deposit(deposition_id):
        """Publish deposit (obtain DOI)"""
        params = {"access_token": ZENODO_TOKEN}
        url = f"{ZENODO_URL}/{deposition_id}/actions/publish"
    
        response = requests.post(url, params=params)
    
        if response.status_code == 202:
            result = response.json()
            doi = result['doi']
            print(f"Published successfully! DOI: {doi}")
            return doi
        else:
            print(f"Publication error: {response.status_code}")
            return None
    
    # Usage example
    deposition_id, bucket_url = create_zenodo_deposit(
        title="Materials Database - Experimental Data",
        description="Experimental materials properties dataset",
        creators=[
            {"name": "Hashimoto, Yusuke",
             "affiliation": "Tohoku University"}
        ],
        keywords=["materials science", "database",
                  "properties"]
    )
    
    if deposition_id:
        # Upload file
        upload_file_to_zenodo(bucket_url, "materials.db.gz")
    
        # Publish (obtain DOI)
        doi = publish_zenodo_deposit(deposition_id)
        print(f"\nDOI: https://doi.org/{doi}")
    

* * *

## 4.6 Practical Project

### 4.6.1 Structuring Experimental Data
    
    
    ```mermaid
    flowchart LR
        A[Lab Notebook] --> B[Excel Data]
        B --> C[pandas DataFrame]
        C --> D[Data Cleaning]
        D --> E[Insert into SQLite]
        E --> F[Quality Check]
        F --> G[Migrate to PostgreSQL]
        G --> H[Publish to Zenodo]
        H --> I[Obtain DOI]
    
        style A fill:#e3f2fd
        style I fill:#e8f5e9
    ```

* * *

## 4.7 Chapter Summary

### What You Learned

  1. **Database Design** \- Schema design using ER diagrams \- Normalization (1NF, 2NF, 3NF) \- Index strategies

  2. **SQLite Implementation** \- CRUD operations (Create, Read, Update, Delete) \- Transaction management \- Query optimization

  3. **PostgreSQL** \- Large-scale data handling \- Bulk insert operations \- Full-text search indexing

  4. **Backup** \- Automated periodic backups \- Compressed storage \- Version control with Git LFS

  5. **Data Publication** \- Upload via Zenodo API \- DOI acquisition \- Open license selection

### Key Points

  * ✅ Schema design is critical from the start
  * ✅ SQLite is optimal for small to medium-scale data
  * ✅ PostgreSQL handles large-scale data
  * ✅ Backups are essential (3-2-1 rule)
  * ✅ Publish data with DOI for citability

### Series Completed!

Congratulations! You have completed the Materials Database Fundamentals series.

**Skills Acquired** : \- Utilizing the four major materials databases \- Complete mastery of Materials Project API \- Integration and cleaning of multiple databases \- Building and publishing custom databases

**Next Steps** : \- Learn machine learning with the \- Efficient exploration with [Bayesian Optimization Introduction](<../bayesian-optimization-introduction/index.html>) \- Apply to your own research projects

**[← Back to Series Contents](<./index.html>)**

* * *

## Exercises

### Exercise 1 (Difficulty: easy)

Create a materials database using SQLite and insert the following data.

**Data** : \- TiO2: rutile, density 4.23 g/cm³, band gap 3.0 eV \- ZnO: wurtzite, density 5.61 g/cm³, band gap 3.4 eV

**Requirements** : 1\. Create materials and properties tables 2\. Insert data 3\. Retrieve and display all data

Solution Example
    
    
    import sqlite3
    
    conn = sqlite3.connect('materials_practice.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE materials (
        material_id INTEGER PRIMARY KEY,
        formula TEXT,
        structure_type TEXT,
        density REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE properties (
        property_id INTEGER PRIMARY KEY,
        material_id INTEGER,
        property_type TEXT,
        value REAL,
        unit TEXT,
        FOREIGN KEY (material_id) REFERENCES materials(material_id)
    )
    ''')
    
    # Insert data
    materials_data = [
        ('TiO2', 'rutile', 4.23),
        ('ZnO', 'wurtzite', 5.61)
    ]
    
    for formula, struct, density in materials_data:
        cursor.execute('''
        INSERT INTO materials (formula, structure_type, density)
        VALUES (?, ?, ?)
        ''', (formula, struct, density))
    
    # Add band gaps
    cursor.execute('''
    INSERT INTO properties (material_id, property_type, value, unit)
    VALUES (1, 'band_gap', 3.0, 'eV')
    ''')
    
    cursor.execute('''
    INSERT INTO properties (material_id, property_type, value, unit)
    VALUES (2, 'band_gap', 3.4, 'eV')
    ''')
    
    conn.commit()
    
    # Retrieve data
    cursor.execute('''
    SELECT m.formula, m.structure_type, m.density, p.value
    FROM materials m
    JOIN properties p ON m.material_id = p.material_id
    WHERE p.property_type = 'band_gap'
    ''')
    
    print("=== Database Contents ===")
    for row in cursor.fetchall():
        print(f"{row[0]}: {row[1]}, {row[2]} g/cm³, "
              f"Eg={row[3]} eV")
    
    conn.close()
    

* * *

### Exercise 2 (Difficulty: medium)

Build a backup system.

**Requirements** : 1\. Daily backups (compressed) 2\. Keep only the latest 5 generations 3\. Backup list display function

Solution Example
    
    
    import os
    import shutil
    import gzip
    from datetime import datetime
    
    class BackupManager:
        def __init__(self, db_path, backup_dir="backups",
                     max_backups=5):
            self.db_path = db_path
            self.backup_dir = backup_dir
            self.max_backups = max_backups
            os.makedirs(backup_dir, exist_ok=True)
    
        def create_backup(self):
            """Create backup"""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}.db.gz"
            backup_path = os.path.join(self.backup_dir,
                                       backup_name)
    
            # Compressed backup
            with open(self.db_path, 'rb') as f_in:
                with gzip.open(backup_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
            print(f"Backup created: {backup_name}")
    
            # Delete old backups
            self.cleanup_old_backups()
    
        def cleanup_old_backups(self):
            """Delete old backups"""
            backups = sorted(
                [f for f in os.listdir(self.backup_dir)
                 if f.endswith('.db.gz')]
            )
    
            if len(backups) > self.max_backups:
                to_delete = backups[:-self.max_backups]
                for backup in to_delete:
                    os.remove(
                        os.path.join(self.backup_dir, backup)
                    )
                    print(f"Deleted: {backup}")
    
        def list_backups(self):
            """List backups"""
            backups = sorted(
                [f for f in os.listdir(self.backup_dir)
                 if f.endswith('.db.gz')]
            )
    
            print("=== Backup List ===")
            for backup in backups:
                size = os.path.getsize(
                    os.path.join(self.backup_dir, backup)
                ) / 1024
                print(f"{backup} ({size:.1f} KB)")
    
    # Usage example
    backup_mgr = BackupManager('materials_practice.db',
                               max_backups=5)
    backup_mgr.create_backup()
    backup_mgr.list_backups()
    

* * *

### Exercise 3 (Difficulty: hard)

Build a complete workflow that stores experimental data in SQLite and publishes it to Zenodo.

**Requirements** : 1\. Load experimental data from Excel file 2\. Store in SQLite 3\. Data quality check 4\. Create backup 5\. Prepare Zenodo metadata (publishing optional)

Solution Example
    
    
    # Requirements:
    # - Python 3.9+
    # - pandas>=2.0.0, <2.2.0
    
    import pandas as pd
    import sqlite3
    import json
    from datetime import datetime
    
    class ExperimentalDataPipeline:
        """Experimental data publication pipeline"""
    
        def __init__(self, db_path='experimental_data.db'):
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path)
            self.setup_database()
    
        def setup_database(self):
            """Database initialization"""
            cursor = self.conn.cursor()
    
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY,
                material_formula TEXT,
                experiment_type TEXT,
                date DATE,
                operator TEXT
            )
            ''')
    
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS measurements (
                measurement_id INTEGER PRIMARY KEY,
                experiment_id INTEGER,
                property_type TEXT,
                value REAL,
                uncertainty REAL,
                unit TEXT,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
            )
            ''')
    
            self.conn.commit()
    
        def load_excel_data(self, file_path):
            """Load data from Excel"""
            df = pd.read_excel(file_path)
            print(f"Data loaded: {len(df)} records")
            return df
    
        def insert_data(self, df):
            """Insert data"""
            cursor = self.conn.cursor()
    
            for _, row in df.iterrows():
                # Insert experiment data
                cursor.execute('''
                INSERT INTO experiments
                (material_formula, experiment_type, date, operator)
                VALUES (?, ?, ?, ?)
                ''', (
                    row['formula'],
                    row['experiment_type'],
                    row['date'],
                    row['operator']
                ))
    
                exp_id = cursor.lastrowid
    
                # Insert measurement values
                cursor.execute('''
                INSERT INTO measurements
                (experiment_id, property_type, value,
                 uncertainty, unit)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    exp_id,
                    row['property'],
                    row['value'],
                    row.get('uncertainty'),
                    row['unit']
                ))
    
            self.conn.commit()
            print(f"{len(df)} records inserted")
    
        def quality_check(self):
            """Quality check"""
            cursor = self.conn.cursor()
    
            # Data counts
            cursor.execute('SELECT COUNT(*) FROM experiments')
            exp_count = cursor.fetchone()[0]
    
            cursor.execute('SELECT COUNT(*) FROM measurements')
            meas_count = cursor.fetchone()[0]
    
            # Check for missing values
            cursor.execute('''
            SELECT COUNT(*) FROM measurements
            WHERE uncertainty IS NULL
            ''')
            missing_unc = cursor.fetchone()[0]
    
            print("=== Quality Report ===")
            print(f"Experiments: {exp_count}")
            print(f"Measurements: {meas_count}")
            print(f"Missing uncertainty: {missing_unc}")
    
        def prepare_zenodo_metadata(self, output_file='zenodo_metadata.json'):
            """Prepare Zenodo metadata"""
            metadata = {
                "title": "Experimental Materials Database",
                "upload_type": "dataset",
                "description": "Experimental data from lab measurements",
                "creators": [
                    {
                        "name": "Your Name",
                        "affiliation": "Your Institution"
                    }
                ],
                "keywords": [
                    "materials science",
                    "experimental data",
                    "properties"
                ],
                "access_right": "open",
                "license": "cc-by-4.0",
                "version": "1.0",
                "publication_date": datetime.now().strftime("%Y-%m-%d")
            }
    
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
            print(f"Metadata saved: {output_file}")
    
        def run(self, excel_file):
            """Run pipeline"""
            print("=== Experimental Data Publication Pipeline ===")
    
            # Load data
            df = self.load_excel_data(excel_file)
    
            # Insert data
            self.insert_data(df)
    
            # Quality check
            self.quality_check()
    
            # Backup
            backup_mgr = BackupManager(self.db_path)
            backup_mgr.create_backup()
    
            # Prepare Zenodo metadata
            self.prepare_zenodo_metadata()
    
            print("\n=== Completed ===")
            print("Next steps:")
            print("1. Verify database")
            print("2. Upload to Zenodo")
            print("3. Obtain DOI")
    
    # Usage example (requires Excel file)
    # pipeline = ExperimentalDataPipeline()
    # pipeline.run('experimental_data.xlsx')
    

* * *

## References

  1. Wilkinson, M. D. et al. (2016). "The FAIR Guiding Principles." _Scientific Data_ , 3, 160018. DOI: [10.1038/sdata.2016.18](<https://doi.org/10.1038/sdata.2016.18>)

  2. Zenodo Documentation. "Developers." URL: [developers.zenodo.org](<https://developers.zenodo.org>)

  3. SQLite Documentation. URL: [sqlite.org/docs](<https://sqlite.org/docs.html>)

  4. PostgreSQL Documentation. URL: [postgresql.org/docs](<https://www.postgresql.org/docs/>)

* * *

## Navigation

### Previous Chapter

**[← Chapter 3: Database Integration and Workflows](<chapter-3.html>)**

### Series Contents

**[← Back to Series Contents](<./index.html>)**

* * *

## Author Information

**Author** : AI Terakoya Content Team **Created** : 2025-10-17 **Version** : 1.0

**License** : Creative Commons BY 4.0

* * *

**Congratulations on completing the Materials Database Fundamentals series!**

**Next Steps** : Dive into machine learning with the
