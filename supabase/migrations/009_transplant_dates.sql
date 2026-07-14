-- supabase/migrations/009_transplant_dates.sql

create table if not exists transplant_dates (
  greenhouse_id int         primary key,
  fecha         date        not null,
  updated_at    timestamptz not null default now()
);

alter table predictions add column if not exists season text;
