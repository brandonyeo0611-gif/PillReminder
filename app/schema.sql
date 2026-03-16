CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    medical_conditions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE medications (
    medication_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    medication_name TEXT,
    dosage TEXT,
    instructions TEXT,
    label_image_url TEXT,
    extracted_label_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE medication_schedules (
    schedule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    medication_id UUID REFERENCES medications(medication_id) ON DELETE CASCADE,
    time_of_day TIME,
    frequency TEXT,
    start_date DATE,
    end_date DATE,
    reminder_enabled BOOLEAN DEFAULT TRUE
);

CREATE TABLE pill_intake_logs (
    intake_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    medication_id UUID REFERENCES medications(medication_id) ON DELETE CASCADE,
    intake_time TIMESTAMP,
    detection_method TEXT,
    confidence_score FLOAT,
    label_scanned BOOLEAN DEFAULT FALSE
);

CREATE TABLE label_scans (
    scan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    medication_id UUID REFERENCES medications(medication_id) ON DELETE CASCADE,
    scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_url TEXT,
    ocr_text TEXT
);

CREATE TABLE ai_food_advice (
    advice_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    illness_id UUID REFERENCES medications(medication_id) ON DELETE CASCADE,
    recommended_foods TEXT,
    recommeneded_lifestyle TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE reminder_logs (
    reminder_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schedule_id UUID REFERENCES medication_schedules(schedule_id) ON DELETE CASCADE,
    reminder_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE
);

-- Note: The instruction asked for these indexes
CREATE INDEX idx_medications_user_id ON medications(user_id);
-- Wait, the instruction specifically said: "Create indexes on: medication_id, user_id, intake_time, schedule_id"
-- medication_id index is already PK on medications, but we should create it on the foreign key columns:
CREATE INDEX idx_medication_schedules_med_id ON medication_schedules(medication_id);
CREATE INDEX idx_pill_intake_user_id ON pill_intake_logs(user_id);
CREATE INDEX idx_pill_intake_med_id ON pill_intake_logs(medication_id);
CREATE INDEX idx_pill_intake_time ON pill_intake_logs(intake_time);
CREATE INDEX idx_label_scans_med_id ON label_scans(medication_id);
CREATE INDEX idx_ai_advice_illness_id ON ai_food_advice(illness_id);
CREATE INDEX idx_reminder_logs_schedule_id ON reminder_logs(schedule_id);
