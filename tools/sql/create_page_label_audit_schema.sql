/*
Second-pass human verification for page classification labels.
Only first-pass REVIEWED target pages are seeded by the API service.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.annotation_pages', N'U') IS NULL
BEGIN
    THROW 50001, N'dbo.annotation_pages does not exist. Initialize annotation tables first.', 1;
END;
GO

IF OBJECT_ID(N'dbo.page_label_audits', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.page_label_audits
    (
        id                 NVARCHAR(36)  NOT NULL,
        page_id            NVARCHAR(36)  NOT NULL,
        original_label     NVARCHAR(64)  NOT NULL,
        audited_label      NVARCHAR(64)  NULL,
        audit_status       NVARCHAR(24)  NOT NULL
            CONSTRAINT DF_page_label_audit_status DEFAULT N'PENDING',
        is_active          BIT           NOT NULL
            CONSTRAINT DF_page_label_audit_is_active DEFAULT (1),
        source_reviewed_at DATETIMEOFFSET(7) NULL,
        audited_by         NVARCHAR(100) NULL,
        audit_note         NVARCHAR(MAX) NULL,
        audited_at         DATETIMEOFFSET(7) NULL,
        created_at         DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_page_label_audit_created_at DEFAULT SYSUTCDATETIME(),
        updated_at         DATETIMEOFFSET(7) NOT NULL
            CONSTRAINT DF_page_label_audit_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT PK_page_label_audits PRIMARY KEY (id),
        CONSTRAINT FK_page_label_audits_page FOREIGN KEY (page_id)
            REFERENCES dbo.annotation_pages(id) ON DELETE CASCADE,
        CONSTRAINT CK_page_label_audit_status CHECK
            (audit_status IN (N'PENDING', N'REVIEWED')),
        CONSTRAINT CK_page_label_audit_original_label CHECK
            (original_label IN
                (N'DIPLOMATIC_NOTE', N'APPLICANT_TABLE', N'APPLICANT_LIST',
                 N'VISA_APPLICATION_FORM', N'TEMPORARY_RESIDENCE_EXTENSION_FORM',
                 N'PASSPORT_FRONT', N'VISA_STICKER',
                 N'VISA_REQUEST_OFFICIAL_LETTER', N'OTHER')),
        CONSTRAINT CK_page_label_audit_final_label CHECK
            (audited_label IS NULL OR audited_label IN
                (N'DIPLOMATIC_NOTE', N'APPLICANT_TABLE', N'APPLICANT_LIST',
                 N'VISA_APPLICATION_FORM', N'TEMPORARY_RESIDENCE_EXTENSION_FORM',
                 N'PASSPORT_FRONT', N'VISA_STICKER',
                 N'VISA_REQUEST_OFFICIAL_LETTER', N'OTHER')),
        CONSTRAINT CK_page_label_audit_reviewed_fields CHECK
            ((audit_status = N'PENDING' AND audited_label IS NULL)
             OR
             (audit_status = N'REVIEWED' AND audited_label IS NOT NULL
              AND audited_by IS NOT NULL AND audited_at IS NOT NULL))
    );
END;
GO

/* Expand the original-label constraint for databases created by an earlier
   version where only the three target labels were allowed. */
IF EXISTS
(
    SELECT 1 FROM sys.check_constraints
    WHERE name = N'CK_page_label_audit_original_label'
      AND parent_object_id = OBJECT_ID(N'dbo.page_label_audits')
      AND definition NOT LIKE N'%OTHER%'
)
BEGIN
    ALTER TABLE dbo.page_label_audits
        DROP CONSTRAINT CK_page_label_audit_original_label;
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.check_constraints
    WHERE name = N'CK_page_label_audit_original_label'
      AND parent_object_id = OBJECT_ID(N'dbo.page_label_audits')
)
BEGIN
    ALTER TABLE dbo.page_label_audits WITH CHECK
        ADD CONSTRAINT CK_page_label_audit_original_label CHECK
        (original_label IN
            (N'DIPLOMATIC_NOTE', N'APPLICANT_TABLE', N'APPLICANT_LIST',
             N'VISA_APPLICATION_FORM', N'TEMPORARY_RESIDENCE_EXTENSION_FORM',
             N'PASSPORT_FRONT', N'VISA_STICKER',
             N'VISA_REQUEST_OFFICIAL_LETTER', N'OTHER'));
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_page_label_audits_active_page'
      AND object_id = OBJECT_ID(N'dbo.page_label_audits')
)
BEGIN
    CREATE UNIQUE INDEX UX_page_label_audits_active_page
        ON dbo.page_label_audits (page_id)
        WHERE is_active = 1;
END;
GO

IF NOT EXISTS
(
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_page_label_audits_queue'
      AND object_id = OBJECT_ID(N'dbo.page_label_audits')
)
BEGIN
    CREATE INDEX IX_page_label_audits_queue
        ON dbo.page_label_audits (is_active, audit_status, created_at)
        INCLUDE (page_id, original_label, audited_label);
END;
GO

PRINT N'Page label audit schema is ready.';
GO
