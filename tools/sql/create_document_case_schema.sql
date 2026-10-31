/*
Group reviewed classified pages into logical diplomatic application cases.
This migration is idempotent and never removes existing data.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.annotation_documents', N'U') IS NULL
   OR OBJECT_ID(N'dbo.annotation_pages', N'U') IS NULL
BEGIN
    THROW 50001, N'Annotation tables do not exist. Initialize them first.', 1;
END;
GO

IF OBJECT_ID(N'dbo.document_cases', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.document_cases
    (
        id                NVARCHAR(36) NOT NULL,
        document_id       NVARCHAR(36) NOT NULL,
        case_index        INT          NOT NULL,
        status            NVARCHAR(24) NOT NULL,
        is_active         BIT          NOT NULL
            CONSTRAINT DF_document_cases_is_active DEFAULT (1),
        start_page_number INT          NOT NULL,
        end_page_number   INT          NOT NULL,
        created_at        DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_document_cases_created_at DEFAULT SYSUTCDATETIME(),
        updated_at        DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_document_cases_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_document_cases PRIMARY KEY (id),
        CONSTRAINT FK_document_cases_document FOREIGN KEY (document_id)
            REFERENCES dbo.annotation_documents(id) ON DELETE CASCADE,
        CONSTRAINT CK_document_cases_index CHECK (case_index >= 1),
        CONSTRAINT CK_document_cases_status CHECK
            (status IN (N'AUTO_GROUPED', N'NEEDS_REVIEW', N'REVIEWED')),
        CONSTRAINT CK_document_cases_page_range CHECK
            (start_page_number >= 1 AND end_page_number >= start_page_number)
    );
END;
GO

IF OBJECT_ID(N'dbo.document_case_pages', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.document_case_pages
    (
        id         NVARCHAR(36) NOT NULL,
        case_id    NVARCHAR(36) NOT NULL,
        page_id    NVARCHAR(36) NOT NULL,
        page_order INT          NOT NULL,
        page_role  NVARCHAR(24) NOT NULL,
        created_at DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_document_case_pages_created_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_document_case_pages PRIMARY KEY (id),
        CONSTRAINT FK_document_case_pages_case FOREIGN KEY (case_id)
            REFERENCES dbo.document_cases(id) ON DELETE CASCADE,
        CONSTRAINT FK_document_case_pages_page FOREIGN KEY (page_id)
            REFERENCES dbo.annotation_pages(id),
        CONSTRAINT UQ_document_case_page UNIQUE (case_id, page_id),
        CONSTRAINT UQ_document_case_page_order UNIQUE (case_id, page_order),
        CONSTRAINT CK_document_case_page_order CHECK (page_order >= 1),
        CONSTRAINT CK_document_case_page_role CHECK
            (page_role IN (N'NOTE', N'APPLICANT_DATA'))
    );
END;
GO

IF OBJECT_ID(N'dbo.case_extraction_annotations', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.case_extraction_annotations
    (
        id            NVARCHAR(36)  NOT NULL,
        case_id       NVARCHAR(36)  NOT NULL,
        schema_version NVARCHAR(100) NOT NULL,
        target_json   NVARCHAR(MAX) NULL,
        review_status NVARCHAR(24)  NOT NULL
            CONSTRAINT DF_case_extraction_review_status DEFAULT N'NEEDS_LABEL',
        reviewed_by   NVARCHAR(100) NULL,
        reviewed_at   DATETIMEOFFSET(7) NULL,
        created_at    DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_case_extraction_created_at DEFAULT SYSUTCDATETIME(),
        updated_at    DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_case_extraction_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_case_extraction_annotations PRIMARY KEY (id),
        CONSTRAINT FK_case_extraction_case FOREIGN KEY (case_id)
            REFERENCES dbo.document_cases(id) ON DELETE CASCADE,
        CONSTRAINT CK_case_extraction_review_status CHECK
            (review_status IN (N'NEEDS_LABEL', N'REVIEWED')),
        CONSTRAINT CK_case_extraction_target_json CHECK
            (target_json IS NULL OR ISJSON(target_json) = 1)
    );
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_document_cases_active_index'
      AND object_id = OBJECT_ID(N'dbo.document_cases')
)
BEGIN
    CREATE UNIQUE INDEX UX_document_cases_active_index
        ON dbo.document_cases (document_id, case_index)
        WHERE is_active = 1;
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_document_cases_queue'
      AND object_id = OBJECT_ID(N'dbo.document_cases')
)
BEGIN
    CREATE INDEX IX_document_cases_queue
        ON dbo.document_cases (is_active, status, document_id, case_index)
        INCLUDE (start_page_number, end_page_number);
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_document_case_pages_page'
      AND object_id = OBJECT_ID(N'dbo.document_case_pages')
)
BEGIN
    CREATE INDEX IX_document_case_pages_page
        ON dbo.document_case_pages (page_id, case_id);
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_case_extraction_review_queue'
      AND object_id = OBJECT_ID(N'dbo.case_extraction_annotations')
)
BEGIN
    CREATE INDEX IX_case_extraction_review_queue
        ON dbo.case_extraction_annotations (review_status, case_id, created_at);
END;
GO

PRINT N'Document case schema is ready.';
GO
