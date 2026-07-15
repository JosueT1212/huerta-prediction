-- supabase/migrations/010_harvest_start_dates.sql

create table if not exists harvest_start_dates (
  greenhouse_id int         primary key,
  fecha         date        not null,
  updated_at    timestamptz not null default now()
);
