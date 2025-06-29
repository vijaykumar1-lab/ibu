

-- Use database
USE health_insurance_ai;




select*from users;
INSERT INTO users (username, password_hash, role)
VALUES 
('vijaykumar', SHA2('123', 256), 'provider'),
('nanditha', SHA2('123', 256), 'admin'),
('dhanush', SHA2('123', 256), 'coder'),
('narmadha', SHA2('123', 256), 'coder');
show tables;
select * from audit_logs;
use heath_insurance_ai;
show tables;
select * from users;
INSERT INTO audit_logs (user_id, action, input_data)
VALUES
(1, 'Login', 'User logged in'),
(2, 'Disease Prediction', 'Vitals: HR=80, SBP=120, DBP=80'),
(3, 'ICD Code Update', 'Assigned ICD: E11 for diabetes'),
(4, 'Viewed Claim History', 'Accessed patient_id: 107');
SELECT user_id, username, role, created_at FROM users;
drop table audit_logs;
use health_insurance_ai;
CREATE TABLE audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    action VARCHAR(255),
    input_data TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
INSERT INTO audit_logs (user_id, action, input_data)
VALUES
(1, 'Login', 'User logged in'),
(2, 'Disease Prediction', 'Vitals: HR=80, SBP=120, DBP=80'),
(3, 'ICD Code Update', 'Assigned ICD: E11 for diabetes'),
(4, 'Viewed Claim History', 'Accessed patient_id: 107');
ALTER TABLE audit_logs ADD COLUMN username VARCHAR(100);
drop table audit_logs;
CREATE TABLE IF NOT EXISTS audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    username VARCHAR(100),
    action VARCHAR(255),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
use health_insurance_ai;
select * from users;



