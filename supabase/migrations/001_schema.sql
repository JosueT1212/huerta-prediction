-- profiles: one row per user, no roles
create table if not exists profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  full_name   text,
  disabled    boolean     not null default false,
  created_at  timestamptz not null default now()
);

-- sensor_readings: time-series archive
create table if not exists sensor_readings (
  id            bigserial   primary key,
  greenhouse_id int         not null,
  recorded_at   timestamptz not null,
  sensor_name   text        not null,
  value         float8      not null
);

create index if not exists sensor_readings_gh_time
  on sensor_readings (greenhouse_id, recorded_at desc);
