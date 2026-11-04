/*
Teacher-generated compact JSON annotations built from versioned text
reconstructions.  This migration is idempotent and never deletes source OCR,
reconstruction or teacher history.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF DB_NAME() <> N'anotation_2'
BEGIN
    THROW 50001, N'Teacher extraction tables must be created in anotation_2.', 1;
END;
GO

IF OBJECT_ID(N'dbo.case_text_reconstructions', N'U') IS NULL
BEGIN
    THROW 50002, N'Text reconstruction tables do not exist. Initialize them first.', 1;
END;
GO

IF OBJECT_ID(N'dbo.teacher_extraction_jobs', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.teacher_extraction_jobs
    (
        id                    VARCHAR(36)   NOT NULL,
        reconstruction_id     VARCHAR(36)   NOT NULL,
        case_id               VARCHAR(36)   NOT NULL,
        document_id           VARCHAR(36)   NOT NULL,
        input_hash            VARCHAR(64)   NOT NULL,
        input_text            NVARCHAR(MAX) NOT NULL,
        source_structure_json NVARCHAR(MAX) NOT NULL,
        reconstruction_version NVARCHAR(100) NOT NULL,
        schema_version        NVARCHAR(100) NOT NULL,
        prompt_version        NVARCHAR(100) NOT NULL,
        teacher_model         NVARCHAR(255) NOT NULL,
        teacher_endpoint      NVARCHAR(500) NOT NULL,
        status                NVARCHAR(20)  NOT NULL
            CONSTRAINT DF_teacher_extraction_job_status DEFAULT N'PENDING',
        is_active             BIT           NOT NULL
            CONSTRAINT DF_teacher_extraction_job_active DEFAULT (1),
        attempt_count         INT           NOT NULL
            CONSTRAINT DF_teacher_extraction_job_attempt_count DEFAULT (0),
        last_error            NVARCHAR(MAX) NULL,
        completed_at          DATETIMEOFFSET(7) NULL,
        created_at            DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_teacher_extraction_job_created DEFAULT SYSUTCDATETIME(),
        updated_at            DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_teacher_extraction_job_updated DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_teacher_extraction_jobs PRIMARY KEY (id),
        CONSTRAINT FK_teacher_extraction_job_reconstruction FOREIGN KEY (reconstruction_id)
            REFERENCES dbo.case_text_reconstructions(id),
        CONSTRAINT FK_teacher_extraction_job_case FOREIGN KEY (case_id)
            REFERENCES dbo.document_cases(id),
        CONSTRAINT FK_teacher_extraction_job_document FOREIGN KEY (document_id)
            REFERENCES dbo.annotation_documents(id),
        CONSTRAINT UQ_teacher_extraction_job_input_hash UNIQUE (input_hash),
        CONSTRAINT CK_teacher_extraction_job_status CHECK
            (status IN (N'PENDING', N'RUNNING', N'COMPLETED', N'FAILED')),
        CONSTRAINT CK_teacher_extraction_job_attempt_count CHECK (attempt_count >= 0),
        CONSTRAINT CK_teacher_extraction_job_source_json CHECK
            (ISJSON(source_structure_json) = 1)
    );
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_teacher_extraction_job_active_case'
      AND object_id = OBJECT_ID(N'dbo.teacher_extraction_jobs')
)
BEGIN
    CREATE UNIQUE INDEX UX_teacher_extraction_job_active_case
        ON dbo.teacher_extraction_jobs (case_id)
        WHERE is_active = 1;
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_teacher_extraction_job_queue'
      AND object_id = OBJECT_ID(N'dbo.teacher_extraction_jobs')
)
BEGIN
    CREATE INDEX IX_teacher_extraction_job_queue
        ON dbo.teacher_extraction_jobs (status, is_active, created_at)
        INCLUDE (case_id, document_id, reconstruction_id, attempt_count);
END;
GO

IF OBJECT_ID(N'dbo.teacher_extraction_attempts', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.teacher_extraction_attempts
    (
        id                     VARCHAR(36)   NOT NULL,
        job_id                 VARCHAR(36)   NOT NULL,
        attempt_number         INT           NOT NULL,
        request_json           NVARCHAR(MAX) NOT NULL,
        raw_response_json      NVARCHAR(MAX) NULL,
        assistant_content      NVARCHAR(MAX) NULL,
        parsed_output_json     NVARCHAR(MAX) NULL,
        validation_status      NVARCHAR(24)  NOT NULL,
        validation_errors_json NVARCHAR(MAX) NULL,
        http_status            INT           NULL,
        latency_ms             INT           NULL,
        prompt_tokens          INT           NULL,
        completion_tokens      INT           NULL,
        total_tokens           INT           NULL,
        finish_reason          NVARCHAR(64)  NULL,
        error_message          NVARCHAR(MAX) NULL,
        created_at             DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_teacher_extraction_attempt_created DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_teacher_extraction_attempts PRIMARY KEY (id),
        CONSTRAINT FK_teacher_extraction_attempt_job FOREIGN KEY (job_id)
            REFERENCES dbo.teacher_extraction_jobs(id) ON DELETE CASCADE,
        CONSTRAINT UQ_teacher_extraction_attempt_number UNIQUE (job_id, attempt_number),
        CONSTRAINT CK_teacher_extraction_attempt_number CHECK (attempt_number >= 1),
        CONSTRAINT CK_teacher_extraction_attempt_validation CHECK
            (validation_status IN
                (N'VALID', N'INVALID', N'PARSE_ERROR', N'HTTP_ERROR', N'TRANSPORT_ERROR')),
        CONSTRAINT CK_teacher_extraction_attempt_http CHECK
            (http_status IS NULL OR (http_status >= 100 AND http_status <= 599)),
        CONSTRAINT CK_teacher_extraction_attempt_latency CHECK
            (latency_ms IS NULL OR latency_ms >= 0),
        CONSTRAINT CK_teacher_extraction_attempt_request_json CHECK
            (ISJSON(request_json) = 1),
        CONSTRAINT CK_teacher_extraction_attempt_response_json CHECK
            (raw_response_json IS NULL OR ISJSON(raw_response_json) = 1),
        CONSTRAINT CK_teacher_extraction_attempt_output_json CHECK
            (parsed_output_json IS NULL OR ISJSON(parsed_output_json) = 1),
        CONSTRAINT CK_teacher_extraction_attempt_errors_json CHECK
            (validation_errors_json IS NULL OR ISJSON(validation_errors_json) = 1)
    );
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_teacher_extraction_attempt_job_created'
      AND object_id = OBJECT_ID(N'dbo.teacher_extraction_attempts')
)
BEGIN
    CREATE INDEX IX_teacher_extraction_attempt_job_created
        ON dbo.teacher_extraction_attempts (job_id, created_at DESC);
END;
GO

IF OBJECT_ID(N'dbo.teacher_extraction_reviews', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.teacher_extraction_reviews
    (
        id                   VARCHAR(36)   NOT NULL,
        job_id               VARCHAR(36)   NOT NULL,
        source_attempt_id    VARCHAR(36)   NOT NULL,
        parent_review_id     VARCHAR(36)   NULL,
        reviewed_output_json NVARCHAR(MAX) NOT NULL,
        review_status        NVARCHAR(20)  NOT NULL
            CONSTRAINT DF_teacher_extraction_review_status DEFAULT N'PENDING',
        is_active            BIT           NOT NULL
            CONSTRAINT DF_teacher_extraction_review_active DEFAULT (1),
        reviewed_by          NVARCHAR(100) NULL,
        review_note          NVARCHAR(MAX) NULL,
        reviewed_at          DATETIMEOFFSET(7) NULL,
        created_at           DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_teacher_extraction_review_created DEFAULT SYSUTCDATETIME(),
        updated_at           DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_teacher_extraction_review_updated DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_teacher_extraction_reviews PRIMARY KEY (id),
        CONSTRAINT FK_teacher_extraction_review_job FOREIGN KEY (job_id)
            REFERENCES dbo.teacher_extraction_jobs(id) ON DELETE CASCADE,
        CONSTRAINT FK_teacher_extraction_review_attempt FOREIGN KEY (source_attempt_id)
            REFERENCES dbo.teacher_extraction_attempts(id),
        CONSTRAINT FK_teacher_extraction_review_parent FOREIGN KEY (parent_review_id)
            REFERENCES dbo.teacher_extraction_reviews(id),
        CONSTRAINT CK_teacher_extraction_review_status CHECK
            (review_status IN (N'PENDING', N'APPROVED', N'REJECTED')),
        CONSTRAINT CK_teacher_extraction_review_json CHECK
            (ISJSON(reviewed_output_json) = 1)
    );
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_teacher_extraction_review_active_job'
      AND object_id = OBJECT_ID(N'dbo.teacher_extraction_reviews')
)
BEGIN
    CREATE UNIQUE INDEX UX_teacher_extraction_review_active_job
        ON dbo.teacher_extraction_reviews (job_id)
        WHERE is_active = 1;
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_teacher_extraction_review_queue'
      AND object_id = OBJECT_ID(N'dbo.teacher_extraction_reviews')
)
BEGIN
    CREATE INDEX IX_teacher_extraction_review_queue
        ON dbo.teacher_extraction_reviews (review_status, is_active, created_at)
        INCLUDE (job_id, source_attempt_id);
END;
GO

PRINT N'Teacher extraction schema is ready.';
GO
