#!/usr/bin/env python3
"""
SQLAlchemy ORM and Columnar Data Formats
=========================================
Demonstrates SQLAlchemy 2.0 ORM with relationships, complex queries, and
high-performance columnar data I/O with PyArrow and Parquet.

SQLAlchemy: https://docs.sqlalchemy.org/en/20/
PyArrow:    https://arrow.apache.org/docs/python/
Parquet:    https://parquet.apache.org/
HDF5:       https://docs.h5py.org/
"""

import os
import time
import tempfile

import numpy as np
import pandas as pd

# SQLAlchemy 2.0 ORM
from sqlalchemy import (
    create_engine, select, func, case, and_, or_, desc, text,
    Integer, String, Float, DateTime, Boolean, ForeignKey,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, Session,
)
from typing import Optional
import datetime

# Columnar formats
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import h5py

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(seed=42)

# =============================================================================
# SQLAlchemy 2.0 ORM: Define Models
# =============================================================================
print("=" * 60)
print("SQLAlchemy 2.0 ORM: Schema Definition")
print("=" * 60)


class Base(DeclarativeBase):
    pass


class Department(Base):
    __tablename__ = 'department'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    location: Mapped[Optional[str]] = mapped_column(String(100))

    # Relationship: one department → many employees
    employees: Mapped[list['Employee']] = relationship(
        back_populates='department', cascade='all, delete-orphan'
    )

    def __repr__(self) -> str:
        return f"Department(id={self.id}, name={self.name!r})"


class Employee(Base):
    __tablename__ = 'employee'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_name: Mapped[str] = mapped_column(String(60), nullable=False)
    last_name: Mapped[str] = mapped_column(String(60), nullable=False)
    email: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    salary: Mapped[float] = mapped_column(Float, nullable=False)
    hire_date: Mapped[datetime.date] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    department_id: Mapped[int] = mapped_column(ForeignKey('department.id'), nullable=False)

    department: Mapped['Department'] = relationship(back_populates='employees')
    projects: Mapped[list['ProjectAssignment']] = relationship(
        back_populates='employee', cascade='all, delete-orphan'
    )

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def __repr__(self) -> str:
        return f"Employee(id={self.id}, name={self.full_name!r}, salary={self.salary:.0f})"


class Project(Base):
    __tablename__ = 'project'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    budget: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default='active')  # active, completed, cancelled

    assignments: Mapped[list['ProjectAssignment']] = relationship(
        back_populates='project', cascade='all, delete-orphan'
    )


class ProjectAssignment(Base):
    __tablename__ = 'project_assignment'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey('employee.id'), nullable=False)
    project_id: Mapped[int] = mapped_column(ForeignKey('project.id'), nullable=False)
    role: Mapped[str] = mapped_column(String(60))
    hours_allocated: Mapped[float] = mapped_column(Float, default=0.0)

    employee: Mapped['Employee'] = relationship(back_populates='projects')
    project: Mapped['Project'] = relationship(back_populates='assignments')


print("Models defined: Department, Employee, Project, ProjectAssignment")

# =============================================================================
# Create In-Memory SQLite DB and Populate with Data
# =============================================================================
print("\n" + "=" * 60)
print("Populating Database")
print("=" * 60)

engine = create_engine('sqlite:///:memory:', echo=False)
Base.metadata.create_all(engine)

# Seed data
dept_data = [
    ('Engineering', 'San Francisco'),
    ('Marketing', 'New York'),
    ('Finance', 'Chicago'),
    ('Operations', 'Austin'),
    ('Research', 'Boston'),
]

first_names = ['Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank', 'Grace', 'Henry',
               'Iris', 'Jack', 'Kate', 'Leo', 'Mia', 'Noah', 'Olivia', 'Paul',
               'Quinn', 'Rachel', 'Sam', 'Tara', 'Uma', 'Victor', 'Wendy', 'Xander']
last_names = ['Smith', 'Jones', 'Brown', 'Williams', 'Taylor', 'Davies', 'Evans',
              'Wilson', 'Thomas', 'Roberts', 'Johnson', 'Lewis', 'Walker', 'Hall']

project_names = [
    'Project Apollo', 'Market Expansion', 'Cost Reduction', 'Digital Transformation',
    'Customer Portal', 'Data Platform', 'Mobile App', 'AI Initiative',
]

with Session(engine) as session:
    # Departments
    departments = [Department(name=name, location=loc) for name, loc in dept_data]
    session.add_all(departments)
    session.flush()  # get IDs

    # Employees (50 total)
    employees = []
    base_date = datetime.date(2018, 1, 1)
    for i in range(50):
        dept = rng.choice(departments)
        hire_days = int(rng.integers(0, 365 * 6))
        salary_base = {'Engineering': 110_000, 'Marketing': 85_000, 'Finance': 95_000,
                       'Operations': 80_000, 'Research': 105_000}[dept.name]
        emp = Employee(
            first_name=rng.choice(first_names),
            last_name=rng.choice(last_names),
            email=f"emp{i:03d}@company.com",
            salary=float(rng.normal(salary_base, salary_base * 0.12)),
            hire_date=datetime.datetime.combine(
                base_date + datetime.timedelta(days=hire_days), datetime.time.min
            ),
            is_active=bool(rng.random() > 0.1),  # 90% active
            department_id=dept.id,
        )
        employees.append(emp)
    session.add_all(employees)
    session.flush()

    # Projects
    projects = [
        Project(
            name=name,
            budget=float(rng.uniform(50_000, 500_000)),
            status=rng.choice(['active', 'completed', 'active', 'active']),
        )
        for name in project_names
    ]
    session.add_all(projects)
    session.flush()

    # Assignments: each project gets 3-8 employees
    roles = ['Lead', 'Developer', 'Analyst', 'Designer', 'Manager', 'Tester']
    for project in projects:
        n_assigned = int(rng.integers(3, 9))
        assigned_emps = rng.choice(employees, size=min(n_assigned, len(employees)), replace=False)
        for emp in assigned_emps:
            session.add(ProjectAssignment(
                employee_id=emp.id,
                project_id=project.id,
                role=rng.choice(roles),
                hours_allocated=float(rng.uniform(40, 400)),
            ))

    session.commit()
    print(f"Inserted: {len(departments)} departments, {len(employees)} employees, "
          f"{len(projects)} projects")

# =============================================================================
# Complex ORM Queries
# =============================================================================
print("\n" + "=" * 60)
print("SQLAlchemy ORM Queries")
print("=" * 60)

with Session(engine) as session:

    # --- Query 1: Department headcount and average salary
    print("\n1. Department headcount and average salary:")
    stmt = (
        select(
            Department.name,
            Department.location,
            func.count(Employee.id).label('headcount'),
            func.round(func.avg(Employee.salary), 2).label('avg_salary'),
            func.round(func.max(Employee.salary), 2).label('max_salary'),
        )
        .join(Employee, Department.id == Employee.department_id)
        .where(Employee.is_active == True)
        .group_by(Department.id)
        .order_by(desc('avg_salary'))
    )
    rows = session.execute(stmt).all()
    print(f"  {'Department':<14} {'Location':<15} {'HC':>4} {'Avg Salary':>12} {'Max Salary':>12}")
    print("  " + "-" * 62)
    for row in rows:
        print(f"  {row.name:<14} {row.location:<15} {row.headcount:>4} "
              f"${row.avg_salary:>11,.0f} ${row.max_salary:>11,.0f}")

    # --- Query 2: Top earners per department (window-function-style via subquery)
    print("\n2. Top 2 earners per department:")
    subq = (
        select(
            Employee.id,
            Employee.first_name,
            Employee.last_name,
            Employee.salary,
            Department.name.label('dept_name'),
            func.rank().over(
                partition_by=Department.id,
                order_by=desc(Employee.salary),
            ).label('salary_rank'),
        )
        .join(Department)
        .where(Employee.is_active == True)
        .subquery()
    )
    stmt2 = select(subq).where(subq.c.salary_rank <= 2).order_by(subq.c.dept_name, subq.c.salary_rank)
    for row in session.execute(stmt2).all():
        print(f"  [{row.salary_rank}] {row.dept_name:<14} {row.first_name} {row.last_name:<10} ${row.salary:,.0f}")

    # --- Query 3: Employees on multiple projects
    print("\n3. Employees assigned to 2+ projects (with total hours):")
    stmt3 = (
        select(
            Employee.first_name,
            Employee.last_name,
            Department.name.label('dept'),
            func.count(ProjectAssignment.project_id).label('project_count'),
            func.round(func.sum(ProjectAssignment.hours_allocated), 1).label('total_hours'),
        )
        .join(ProjectAssignment, Employee.id == ProjectAssignment.employee_id)
        .join(Department, Employee.department_id == Department.id)
        .group_by(Employee.id)
        .having(func.count(ProjectAssignment.project_id) >= 2)
        .order_by(desc('project_count'))
        .limit(8)
    )
    for row in session.execute(stmt3).all():
        print(f"  {row.first_name} {row.last_name:<12} ({row.dept:<14}) "
              f"— {row.project_count} projects, {row.total_hours} h")

    # --- Query 4: Project budget vs allocated hours
    print("\n4. Project summary (budget vs allocated effort):")
    stmt4 = (
        select(
            Project.name,
            Project.status,
            func.round(Project.budget, 0).label('budget'),
            func.count(ProjectAssignment.id).label('team_size'),
            func.round(func.sum(ProjectAssignment.hours_allocated), 0).label('total_hours'),
        )
        .join(ProjectAssignment, isouter=True)
        .group_by(Project.id)
        .order_by(desc(Project.budget))
    )
    for row in session.execute(stmt4).all():
        print(f"  {row.name:<25} [{row.status:<10}] "
              f"budget=${row.budget:>8,.0f}  team={row.team_size}  hours={row.total_hours or 0:.0f}")

    # --- Query 5: Conditional salary bands
    print("\n5. Salary distribution by band:")
    salary_band = case(
        (Employee.salary >= 120_000, 'Senior (≥$120k)'),
        (Employee.salary >= 90_000, 'Mid ($90k-$120k)'),
        else_='Junior (<$90k)',
    )
    stmt5 = (
        select(
            salary_band.label('band'),
            func.count(Employee.id).label('count'),
            func.round(func.avg(Employee.salary), 0).label('avg'),
        )
        .group_by(salary_band)
        .order_by(desc('avg'))
    )
    for row in session.execute(stmt5).all():
        print(f"  {row.band:<22}: {row.count:>3} employees, avg ${row.avg:,.0f}")

# =============================================================================
# PyArrow: High-Performance Columnar Data
# =============================================================================
print("\n" + "=" * 60)
print("PyArrow: Schema-First Columnar Data")
print("=" * 60)

# Simulate a large transaction dataset
n_rows = 200_000
schema = pa.schema([
    pa.field('transaction_id', pa.int64()),
    pa.field('customer_id', pa.int32()),
    pa.field('amount', pa.float64()),
    pa.field('currency', pa.dictionary(pa.int8(), pa.string())),
    pa.field('category', pa.dictionary(pa.int8(), pa.string())),
    pa.field('timestamp', pa.timestamp('ms')),
    pa.field('is_fraud', pa.bool_()),
])

currencies = ['USD', 'EUR', 'GBP', 'JPY']
categories = ['Electronics', 'Groceries', 'Travel', 'Dining', 'Healthcare', 'Entertainment']

start_ts = pd.Timestamp('2024-01-01').value // 10**6
end_ts = pd.Timestamp('2024-12-31').value // 10**6

timestamps_ms = rng.integers(start_ts, end_ts, size=n_rows).astype('int64')

amounts = rng.lognormal(mean=3.5, sigma=1.2, size=n_rows)
is_fraud = rng.random(size=n_rows) < 0.02  # 2% fraud rate
amounts[is_fraud] *= rng.uniform(3, 20, size=is_fraud.sum())  # fraud txns are larger

table = pa.table(
    {
        'transaction_id': pa.array(np.arange(n_rows, dtype=np.int64)),
        'customer_id': pa.array(rng.integers(1, 10_001, size=n_rows, dtype=np.int32)),
        'amount': pa.array(amounts),
        'currency': pa.array(rng.choice(currencies, size=n_rows)).dictionary_encode(),
        'category': pa.array(rng.choice(categories, size=n_rows)).dictionary_encode(),
        'timestamp': pa.array(timestamps_ms, type=pa.timestamp('ms')),
        'is_fraud': pa.array(is_fraud),
    },
    schema=schema,
)

print(f"Table: {table.num_rows:,} rows × {table.num_columns} columns")
print(f"Schema:\n{table.schema}")

# PyArrow compute functions — no pandas needed
fraud_count = pc.sum(table.column('is_fraud')).as_py()
fraud_rate = fraud_count / n_rows
total_amount = pc.sum(table.column('amount')).as_py()
avg_amount = pc.mean(table.column('amount')).as_py()

print(f"\nArrow compute results:")
print(f"  Fraud rate         : {fraud_rate:.2%}")
print(f"  Total amount       : ${total_amount:,.2f}")
print(f"  Average amount     : ${avg_amount:.2f}")
print(f"  Max amount         : ${pc.max(table.column('amount')).as_py():.2f}")

# =============================================================================
# Parquet: Write and Read with Row Groups and Filters
# =============================================================================
print("\n" + "=" * 60)
print("Parquet: Partitioned Write and Predicate Pushdown")
print("=" * 60)

parquet_path = os.path.join(OUTPUT_DIR, 'transactions.parquet')

# Write with compression and row group size
t_write_start = time.perf_counter()
pq.write_table(
    table,
    parquet_path,
    compression='snappy',
    row_group_size=50_000,
    write_statistics=True,
)
write_time = time.perf_counter() - t_write_start
file_size_mb = os.path.getsize(parquet_path) / 1e6
print(f"Written: {parquet_path}")
print(f"  Size            : {file_size_mb:.2f} MB  ({n_rows * 8 * 7 / 1e6:.1f} MB uncompressed est.)")
print(f"  Write time      : {write_time*1000:.1f} ms")
print(f"  Compression     : Snappy ({(1 - file_size_mb / (n_rows * 8 * 7 / 1e6)) * 100:.0f}% reduction)")

# Read metadata without loading data
parquet_file = pq.ParquetFile(parquet_path)
meta = parquet_file.metadata
print(f"\nParquet metadata:")
print(f"  Row groups      : {meta.num_row_groups}")
print(f"  Total rows      : {meta.num_rows:,}")
print(f"  Columns         : {meta.num_columns}")
print(f"  Format version  : {meta.format_version}")

# Predicate pushdown — read only fraud transactions using filters
t_read_start = time.perf_counter()
fraud_table = pq.read_table(
    parquet_path,
    columns=['transaction_id', 'customer_id', 'amount', 'category', 'timestamp'],
    filters=[('is_fraud', '=', True)],
)
read_time = time.perf_counter() - t_read_start

print(f"\nPredicate pushdown read (is_fraud=True):")
print(f"  Rows returned   : {fraud_table.num_rows:,} ({fraud_table.num_rows/n_rows:.2%} of total)")
print(f"  Read time       : {read_time*1000:.1f} ms  (only scanned matching row groups)")

fraud_df = fraud_table.to_pandas()
print(f"\nFraud transaction stats:")
print(f"  Amount mean     : ${fraud_df['amount'].mean():.2f}")
print(f"  Amount median   : ${fraud_df['amount'].median():.2f}")
print(f"  Amount max      : ${fraud_df['amount'].max():.2f}")
print(f"  Top categories  : {fraud_df['category'].value_counts().head(3).to_dict()}")

# Column projection — read only needed columns (much faster for wide tables)
t_proj_start = time.perf_counter()
amounts_only = pq.read_table(parquet_path, columns=['amount', 'is_fraud'])
proj_time = time.perf_counter() - t_proj_start
print(f"\nColumn projection (2 of {meta.num_columns} columns): {proj_time*1000:.1f} ms")

# =============================================================================
# HDF5: Hierarchical Data Format
# =============================================================================
print("\n" + "=" * 60)
print("HDF5: Hierarchical Numerical Storage")
print("=" * 60)

hdf5_path = os.path.join(OUTPUT_DIR, 'simulation_data.h5')

# Write simulation results in groups
with h5py.File(hdf5_path, 'w') as f:
    # Metadata as attributes
    f.attrs['created'] = str(datetime.datetime.utcnow())
    f.attrs['author'] = 'DataIO Example'
    f.attrs['version'] = '1.0'

    # Group 1: time series simulation
    sim = f.create_group('simulation')
    sim.attrs['description'] = 'Monte Carlo price simulation'
    n_paths, n_steps = 500, 252
    t_sim = np.linspace(0, 1, n_steps)
    dt = t_sim[1] - t_sim[0]
    mu, sigma_sim = 0.08, 0.20
    W = rng.standard_normal(size=(n_paths, n_steps))
    paths = 100.0 * np.exp(np.cumsum((mu - 0.5 * sigma_sim**2) * dt + sigma_sim * np.sqrt(dt) * W, axis=1))
    ds = sim.create_dataset('paths', data=paths, compression='gzip', compression_opts=4, chunks=True)
    ds.attrs['shape_description'] = f'({n_paths} paths, {n_steps} time steps)'
    sim.create_dataset('time_axis', data=t_sim)

    # Group 2: model parameters
    params = f.create_group('parameters')
    params.create_dataset('mu', data=mu)
    params.create_dataset('sigma', data=sigma_sim)
    params.create_dataset('initial_price', data=100.0)

    # Group 3: statistical summary
    stats_grp = f.create_group('statistics')
    final_prices_arr = paths[:, -1]
    stats_grp.create_dataset('final_prices', data=final_prices_arr)
    stats_grp.attrs['mean_final'] = float(final_prices_arr.mean())
    stats_grp.attrs['std_final'] = float(final_prices_arr.std())
    stats_grp.attrs['var_95'] = float(np.percentile(final_prices_arr, 5))
    stats_grp.attrs['var_99'] = float(np.percentile(final_prices_arr, 1))

print(f"Written: {hdf5_path}  ({os.path.getsize(hdf5_path)/1e6:.2f} MB)")

# Read back and inspect
with h5py.File(hdf5_path, 'r') as f:
    print(f"\nHDF5 structure:")
    def print_hdf5(name: str, obj: h5py.HLObject) -> None:
        indent = "  " + "  " * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}[dataset] /{name}  shape={obj.shape}  dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}[group]   /{name}/")
    f.visititems(print_hdf5)

    s = f['statistics']
    print(f"\nMonte Carlo results ({n_paths} paths, {n_steps} steps):")
    print(f"  Mean final price : ${s.attrs['mean_final']:.2f}")
    print(f"  Std final price  : ${s.attrs['std_final']:.2f}")
    print(f"  VaR 95%          : ${s.attrs['var_95']:.2f}  (5th percentile)")
    print(f"  VaR 99%          : ${s.attrs['var_99']:.2f}  (1st percentile)")

    # Load a slice without reading the full dataset
    first_10_paths = f['simulation/paths'][:10, :]
    print(f"  Slice read (10 paths): {first_10_paths.shape}  (only {first_10_paths.nbytes/1024:.1f} KB loaded)")

# =============================================================================
# Performance Comparison: JSON vs Parquet vs HDF5
# =============================================================================
print("\n" + "=" * 60)
print("I/O Format Performance Comparison")
print("=" * 60)

sample_df = pd.DataFrame({
    'id': np.arange(100_000),
    'x': rng.standard_normal(100_000),
    'y': rng.standard_normal(100_000),
    'category': rng.choice(['A', 'B', 'C', 'D'], size=100_000),
})

results = {}
with tempfile.TemporaryDirectory() as tmpdir:
    for fmt, write_fn, read_fn, ext in [
        ('CSV',     lambda p: sample_df.to_csv(p, index=False),       pd.read_csv,          'csv'),
        ('JSON',    lambda p: sample_df.to_json(p, orient='records'),  pd.read_json,         'json'),
        ('Parquet', lambda p: sample_df.to_parquet(p),                pd.read_parquet,      'parquet'),
        ('Feather', lambda p: sample_df.to_feather(p),                pd.read_feather,      'feather'),
    ]:
        path = os.path.join(tmpdir, f'bench.{ext}')

        t0 = time.perf_counter()
        write_fn(path)
        write_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        _ = read_fn(path)
        read_ms = (time.perf_counter() - t0) * 1000

        size_kb = os.path.getsize(path) / 1024
        results[fmt] = {'write_ms': write_ms, 'read_ms': read_ms, 'size_kb': size_kb}

print(f"  {'Format':<10} {'Write (ms)':>12} {'Read (ms)':>12} {'Size (KB)':>12}")
print("  " + "-" * 50)
for fmt, r in results.items():
    print(f"  {fmt:<10} {r['write_ms']:>12.1f} {r['read_ms']:>12.1f} {r['size_kb']:>12.1f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SQLAlchemy ORM and Columnar Data I/O complete!")
print(f"Outputs: transactions.parquet  ({file_size_mb:.1f} MB)")
print(f"         simulation_data.h5    ({os.path.getsize(hdf5_path)/1e6:.1f} MB)")
print("=" * 60)
