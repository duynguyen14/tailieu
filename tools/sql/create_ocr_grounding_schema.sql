/*
Store OCR text and immutable source coordinates used to ground LLM outputs.
The script is idempotent and never deletes existing OCR runs or tokens.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.annotation_pages', N'U') IS NULL
BEGIN
    THROW 50001, N'dbo.annotation_pages does not exist. Initialize annotation tables first.', 1;
END;
GO

IF OBJECT_ID(N'dbo.ocr_runs', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_runs
    (
        id                 NVARCHAR(36)  NOT NULL,
        page_id            NVARCHAR(36)  NOT NULL,
        status             NVARCHAR(20)  NOT NULL,
        is_active          BIT           NOT NULL
            CONSTRAINT DF_ocr_runs_is_active DEFAULT (1),
        model_name         NVARCHAR(255) NOT NULL,
        model_version      NVARCHAR(100) NOT NULL,
        device             NVARCHAR(32)  NOT NULL,
        processing_time_ms INT           NULL,
        error_message      NVARCHAR(MAX) NULL,
        created_at         DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_runs_created_at DEFAULT SYSUTCDATETIME(),
        updated_at         DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_runs_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_ocr_runs PRIMARY KEY (id),
        CONSTRAINT FK_ocr_runs_page FOREIGN KEY (page_id)
            REFERENCES dbo.annotation_pages(id) ON DELETE CASCADE,
        CONSTRAINT CK_ocr_runs_status CHECK
            (status IN (N'PROCESSING', N'COMPLETED', N'FAILED')),
        CONSTRAINT CK_ocr_runs_processing_time CHECK
            (processing_time_ms IS NULL OR processing_time_ms >= 0)
    );
END;
GO

IF OBJECT_ID(N'dbo.ocr_tokens', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ocr_tokens
    (
        id           NVARCHAR(36)  NOT NULL,
        ocr_run_id   NVARCHAR(36)  NOT NULL,
        token_index  INT           NOT NULL,
        token_id     NVARCHAR(20)  NOT NULL,
        text         NVARCHAR(MAX) NOT NULL,
        confidence   DECIMAL(9,6)  NOT NULL,
        x1           INT           NOT NULL,
        y1           INT           NOT NULL,
        x2           INT           NOT NULL,
        y2           INT           NOT NULL,
        polygon_json NVARCHAR(MAX) NULL,
        created_at   DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_ocr_tokens_created_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_ocr_tokens PRIMARY KEY (id),
        CONSTRAINT FK_ocr_tokens_run FOREIGN KEY (ocr_run_id)
            REFERENCES dbo.ocr_runs(id) ON DELETE CASCADE,
        CONSTRAINT UQ_ocr_tokens_run_index UNIQUE (ocr_run_id, token_index),
        CONSTRAINT UQ_ocr_tokens_run_token_id UNIQUE (ocr_run_id, token_id),
        CONSTRAINT CK_ocr_tokens_index CHECK (token_index >= 1),
        CONSTRAINT CK_ocr_tokens_confidence CHECK
            (confidence >= 0 AND confidence <= 1),
        CONSTRAINT CK_ocr_tokens_bbox CHECK
            (x1 >= 0 AND y1 >= 0 AND x2 > x1 AND y2 > y1)
    );
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_ocr_runs_page_active'
      AND object_id = OBJECT_ID(N'dbo.ocr_runs')
)
BEGIN
    CREATE UNIQUE INDEX UX_ocr_runs_page_active
        ON dbo.ocr_runs (page_id)
        WHERE is_active = 1;
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_ocr_runs_training_export'
      AND object_id = OBJECT_ID(N'dbo.ocr_runs')
)
BEGIN
    CREATE INDEX IX_ocr_runs_training_export
        ON dbo.ocr_runs (status, is_active, created_at)
        INCLUDE (page_id, model_name, model_version, device);
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_ocr_tokens_run_order'
      AND object_id = OBJECT_ID(N'dbo.ocr_tokens')
)
BEGIN
    CREATE INDEX IX_ocr_tokens_run_order
        ON dbo.ocr_tokens (ocr_run_id, token_index)
        INCLUDE (token_id, confidence, x1, y1, x2, y2);
END;
GO

PRINT N'OCR grounding schema is ready.';
GO

