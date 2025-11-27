'use client';

import React from 'react';

type SimplePDFViewerProps = {
  file: string;
  onClose?: () => void;
};

export const PDFViewer: React.FC<SimplePDFViewerProps> = ({ file, onClose }) => {
  return (
    <div className="w-full h-full flex flex-col bg-white">
      <div className="flex justify-between items-center p-4 border-b">
        <h2 className="text-lg font-semibold">PDF Viewer</h2>
        {onClose && (
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded text-sm"
          >
            Close
          </button>
        )}
      </div>
      <div className="flex-1 overflow-auto">
        <iframe
          src={file}
          className="w-full h-full border-none"
          title="PDF Viewer"
        />
      </div>
    </div>
  );
};