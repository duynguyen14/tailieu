/*
Versioned structured-text snapshots built from approved OCR reviews.
This migration is idempotent and never deletes OCR or reconstruction history.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.document_cases', N'U') IS NULL
BEGIN
    THROW 50001, N'Document case tables do not exist. Initialize them first.', 1;
END;
GO

IF OBJECT_ID(N'dbo.case_text_reconstructions', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.case_text_reconstructions
    (
        id                       NVARCHAR(36)  NOT NULL,
        case_id                  NVARCHAR(36)  NOT NULL,
        parent_reconstruction_id NVARCHAR(36)  NULL,
        algorithm_version        NVARCHAR(100) NOT NULL,
        source_fingerprint       NVARCHAR(64)  NOT NULL,
        status                   NVARCHAR(20)  NOT NULL
            CONSTRAINT DF_case_text_reconstruction_status DEFAULT N'GENERATED',
        is_active                BIT           NOT NULL
            CONSTRAINT DF_case_text_reconstruction_active DEFAULT (1),
        reconstructed_text       NVARCHAR(MAX) NOT NULL,
        structure_json           NVARCHAR(MAX) NOT NULL,
        page_count               INT           NOT NULL,
        token_count              INT           NOT NULL,
        reviewed_by              NVARCHAR(100) NULL,
        review_note              NVARCHAR(MAX) NULL,
        reviewed_at              DATETIMEOFFSET(7) NULL,
        created_at               DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_case_text_reconstruction_created DEFAULT SYSUTCDATETIME(),
        updated_at               DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_case_text_reconstruction_updated DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_case_text_reconstructions PRIMARY KEY (id),
        CONSTRAINT FK_case_text_reconstruction_case FOREIGN KEY (case_id)
            REFERENCES dbo.document_cases(id) ON DELETE CASCADE,
        CONSTRAINT FK_case_text_reconstruction_parent FOREIGN KEY (parent_reconstruction_id)
            REFERENCES dbo.case_text_reconstructions(id),
        CONSTRAINT CK_case_text_reconstruction_status CHECK
            (status IN (N'GENERATED', N'REVIEWED')),
        CONSTRAINT CK_case_text_reconstruction_pages CHECK (page_count >= 1),
        CONSTRAINT CK_case_text_reconstruction_tokens CHECK (token_count >= 0),
        CONSTRAINT CK_case_text_reconstruction_json CHECK (ISJSON(structure_json) = 1)
    );
END;
GO

IF COL_LENGTH(N'dbo.case_text_reconstructions', N'reviewed_by') IS NULL
    ALTER TABLE dbo.case_text_reconstructions ADD reviewed_by NVARCHAR(100) NULL;
GO

IF COL_LENGTH(N'dbo.case_text_reconstructions', N'review_note') IS NULL
    ALTER TABLE dbo.case_text_reconstructions ADD review_note NVARCHAR(MAX) NULL;
GO

IF COL_LENGTH(N'dbo.case_text_reconstructions', N'reviewed_at') IS NULL
    ALTER TABLE dbo.case_text_reconstructions ADD reviewed_at DATETIMEOFFSET(7) NULL;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_case_text_reconstruction_active'
      AND object_id = OBJECT_ID(N'dbo.case_text_reconstructions')
)
BEGIN
    CREATE UNIQUE INDEX UX_case_text_reconstruction_active
        ON dbo.case_text_reconstructions (case_id)
        WHERE is_active = 1;
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_case_text_reconstruction_source'
      AND object_id = OBJECT_ID(N'dbo.case_text_reconstructions')
)
BEGIN
    CREATE INDEX IX_case_text_reconstruction_source
        ON dbo.case_text_reconstructions (case_id, source_fingerprint, algorithm_version)
        INCLUDE (status, is_active, created_at);
END;
GO

PRINT N'Text reconstruction schema is ready.';
GO
