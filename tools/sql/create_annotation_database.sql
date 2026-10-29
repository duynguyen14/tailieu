/*
    Document annotation database for Microsoft SQL Server.

    Default database name matches DB_NAME=document_annotation in .env.
    If you change DB_NAME, change [document_annotation] in the two places below.

    This script is safe to run again: existing database/tables/indexes are kept.
*/

USE [master];
GO

IF DB_ID(N'document_annotation') IS NULL
BEGIN
    PRINT N'Creating database [document_annotation]...';
    EXEC(N'CREATE DATABASE [document_annotation]');
END
ELSE
BEGIN
    PRINT N'Database [document_annotation] already exists.';
END;
GO

USE [document_annotation];
GO

/* One row per uploaded PDF document. */
IF OBJECT_ID(N'dbo.annotation_documents', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.annotation_documents
    (
        id                NVARCHAR(36)  NOT NULL,
        original_filename NVARCHAR(255) NOT NULL,

        -- Relative to ANNOTATION_STORAGE_DIR; never store an absolute path.
        pdf_relative_path NVARCHAR(500) NOT NULL,

        -- Used to detect duplicate uploads.
        sha256            NVARCHAR(64)  NOT NULL,
        page_count        INT           NOT NULL,
        review_status     NVARCHAR(32)  NOT NULL
            CONSTRAINT DF_annotation_documents_review_status DEFAULT N'UNLABELED',
        created_at        DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_annotation_documents_created_at DEFAULT SYSUTCDATETIME(),
        updated_at        DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_annotation_documents_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_annotation_documents PRIMARY KEY (id),
        CONSTRAINT UQ_annotation_documents_sha256 UNIQUE (sha256),
        CONSTRAINT CK_annotation_documents_page_count CHECK (page_count >= 1),
        CONSTRAINT CK_annotation_documents_review_status CHECK
            (review_status IN (N'UNLABELED', N'IN_PROGRESS', N'REVIEWED'))
    );

    PRINT N'Created table dbo.annotation_documents.';
END
ELSE
BEGIN
    PRINT N'Table dbo.annotation_documents already exists.';
END;
GO

/* One row per rendered PDF page. */
IF OBJECT_ID(N'dbo.annotation_pages', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.annotation_pages
    (
        id                  NVARCHAR(36)  NOT NULL,
        document_id         NVARCHAR(36)  NOT NULL,
        page_index          INT           NOT NULL,
        page_number         INT           NOT NULL,

        -- Relative to ANNOTATION_STORAGE_DIR; never store an absolute path.
        image_relative_path NVARCHAR(500) NOT NULL,
        mime_type           NVARCHAR(64)  NOT NULL
            CONSTRAINT DF_annotation_pages_mime_type DEFAULT N'image/png',
        width               INT           NOT NULL,
        height              INT           NOT NULL,

        -- Null until a reviewer assigns one of the four page classes.
        label               NVARCHAR(64)  NULL,
        review_status       NVARCHAR(32)  NOT NULL
            CONSTRAINT DF_annotation_pages_review_status DEFAULT N'UNLABELED',
        reviewed_by         NVARCHAR(100) NULL,
        review_note         NVARCHAR(MAX) NULL,
        reviewed_at         DATETIMEOFFSET(7) NULL,
        created_at          DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_annotation_pages_created_at DEFAULT SYSUTCDATETIME(),
        updated_at          DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_annotation_pages_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_annotation_pages PRIMARY KEY (id),
        CONSTRAINT FK_annotation_pages_document FOREIGN KEY (document_id)
            REFERENCES dbo.annotation_documents(id) ON DELETE CASCADE,
        CONSTRAINT UQ_annotation_pages_document_page_index
            UNIQUE (document_id, page_index),
        CONSTRAINT UQ_annotation_pages_document_page_number
            UNIQUE (document_id, page_number),
        CONSTRAINT CK_annotation_pages_page_index CHECK (page_index >= 0),
        CONSTRAINT CK_annotation_pages_page_number CHECK (page_number >= 1),
        CONSTRAINT CK_annotation_pages_dimensions CHECK (width > 0 AND height > 0),
        CONSTRAINT CK_annotation_pages_label CHECK
            (label IS NULL OR label IN
                (N'DIPLOMATIC_NOTE', N'APPLICANT_TABLE', N'APPLICANT_LIST', N'OTHER')),
        CONSTRAINT CK_annotation_pages_review_status CHECK
            (review_status IN (N'UNLABELED', N'REVIEWED'))
    );

    PRINT N'Created table dbo.annotation_pages.';
END
ELSE
BEGIN
    PRINT N'Table dbo.annotation_pages already exists.';
END;
GO

/* Indexes used by the listing and review APIs. */
IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_annotation_documents_review_status'
      AND object_id = OBJECT_ID(N'dbo.annotation_documents')
)
    CREATE INDEX IX_annotation_documents_review_status
        ON dbo.annotation_documents (review_status);
GO

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_annotation_pages_document_id'
      AND object_id = OBJECT_ID(N'dbo.annotation_pages')
)
    CREATE INDEX IX_annotation_pages_document_id
        ON dbo.annotation_pages (document_id);
GO

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_annotation_pages_label'
      AND object_id = OBJECT_ID(N'dbo.annotation_pages')
)
    CREATE INDEX IX_annotation_pages_label
        ON dbo.annotation_pages (label);
GO

IF NOT EXISTS
(
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_annotation_pages_review_status'
      AND object_id = OBJECT_ID(N'dbo.annotation_pages')
)
    CREATE INDEX IX_annotation_pages_review_status
        ON dbo.annotation_pages (review_status);
GO

PRINT N'Database and annotation tables are ready.';
GO
