// BlockingContext.tsx
import React, {
    createContext,
    useContext,
    useState,
    useMemo,
    type PropsWithChildren,
  } from 'react';
  import Backdrop from '@mui/material/Backdrop';
  import CircularProgress from '@mui/material/CircularProgress';
  
  type BlockingValue = {
    blocking: boolean;
    setBlocking: (v: boolean) => void;
  };
  
  const BlockingCtx = createContext<BlockingValue>({
    blocking: false,
    setBlocking: () => {},
  });
  
  export const BlockingProvider: React.FC<PropsWithChildren> = ({ children }) => {
    const [blocking, setBlocking] = useState(false);
    const value = useMemo(() => ({ blocking, setBlocking }), [blocking]);
  
    return (
      <BlockingCtx.Provider value={value}>
        <Backdrop
          open={blocking}
          sx={{
            color: '#fff',
            zIndex: (theme) => theme.zIndex.modal + 2,
            bgcolor: 'rgba(0,0,0,0.35)',
          }}
        >
          <CircularProgress />
        </Backdrop>
        {children}
      </BlockingCtx.Provider>
    );
  };
  
  export const useBlocking = () => useContext(BlockingCtx);
  