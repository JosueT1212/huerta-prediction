-- supabase/migrations/008_split_sensor_tables.sql

-- Sensores (internas) — trim sensor_readings_wide to internas-only, add the
-- 3 previously-missing columns, drop the 4 ext + 3 riego columns the
-- original 003_sensor_wide.sql incorrectly bundled in.
alter table sensor_readings_wide
  add column if not exists deficit_humedad float8,
  add column if not exists deficit_presion_vapor float8,
  add column if not exists humedad_abs_int float8,
  drop column if exists temp_prom_ext,
  drop column if exists temp_max_ext,
  drop column if exists temp_min_ext,
  drop column if exists rad_sum,
  drop column if exists riego_total,
  drop column if exists ph_promedio,
  drop column if exists ce_promedio;

-- Riego — new table, per-invernadero
create table if not exists riego_readings (
  id            bigserial   primary key,
  greenhouse_id int         not null,
  fecha         date        not null,
  riego_total   float8,
  ph_promedio   float8,
  ce_promedio   float8
);
create unique index if not exists riego_readings_gh_date
  on riego_readings (greenhouse_id, fecha);

-- Exteriores — new table, GLOBAL (no greenhouse_id column)
create table if not exists exterior_readings (
  id                bigserial primary key,
  fecha             date      not null,
  temp_prom_ext     float8,
  temp_max_ext      float8,
  temp_min_ext      float8,
  hr_prom_ext       float8,
  rad_sum           float8,
  rad_max           float8,
  dh_ext            float8,
  humedad_abs_ext   float8
);
create unique index if not exists exterior_readings_date
  on exterior_readings (fecha);

-- Extend submission_locks.form_type to cover the 2 new types
alter table submission_locks drop constraint if exists submission_locks_form_type_check;
alter table submission_locks add constraint submission_locks_form_type_check
  check (form_type in ('sensores','riego','exteriores','fenologia','produccion'));
